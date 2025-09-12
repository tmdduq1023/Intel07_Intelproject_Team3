// mainwindow.cpp
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QCameraInfo>
#include <QMessageBox>
#include <QDateTime>
#include <QDebug>
#include <QBuffer>
#include <QHttpMultiPart>
#include <QHttpPart>
#include <QNetworkRequest>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSettings>
#include <QProgressDialog>
#include <QTimer>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QGraphicsEllipseItem>
#include <QGraphicsTextItem>
#include <QPen>
#include <QBrush>
#include <QFont>
#include <QResizeEvent>
#include <QApplication>
#include <QDesktopWidget>
#include <QScreen>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , faceGuideCircle(nullptr)
    , guideTextItem(nullptr)
    , isCameraRunning(false)
    , isNameEntered(false)
{
    ui->setupUi(this);

    // 화면 크기에 맞춰 창을 최대화하고 동적 크기 조정 가능하게 설정
    setupWindowSizing();

    // 레이아웃 설정 (UI 파일에 레이아웃이 없는 경우)
    setupUILayout();

    // 네트워크 매니저 초기화
    networkManager = std::make_unique<QNetworkAccessManager>(this);
    connect(networkManager.get(), &QNetworkAccessManager::finished,
            this, &MainWindow::onUploadFinished);

    // 서버 설정 로드
    loadServerConfig();

    // 데이터베이스 초기화
    initializeDatabase();

    // 초기 화면 설정 (이름 입력 단계)
    setupInitialView();

    // 상태바에 서버 정보 표시
    ui->statusbar->showMessage(QString("Server: %1").arg(serverUrl));
}

MainWindow::~MainWindow()
{
    if (camera && camera->state() == QCamera::ActiveState) {
        camera->stop();
    }
    delete ui;
}

void MainWindow::loadServerConfig()
{
    // QSettings를 사용하여 설정 파일에서 서버 정보 읽기
    // 또는 하드코딩된 값 사용
    QSettings settings("config.ini", QSettings::IniFormat);

    // 기본값 설정 (필요에 따라 수정)
    serverUrl = settings.value("Server/url", "http://192.168.0.90:5000").toString();
    serverEndpoint = settings.value("Server/endpoint", "/upload").toString();
    raspUrl = settings.value("Server/rasp_url", "http://localhost:5000").toString();

    // 설정 파일이 없으면 생성
    if (!settings.contains("Server/url")) {
        settings.setValue("Server/url", serverUrl);
        settings.setValue("Server/endpoint", serverEndpoint);
        settings.setValue("Server/rasp_url", raspUrl);
        settings.sync();

        qDebug() << "Created config.ini with default server settings";
    }

    qDebug() << "Server URL:" << serverUrl + serverEndpoint;
    qDebug() << "Rasp URL:" << raspUrl;
}

void MainWindow::setupWindowSizing()
{
    // 현재 사용 가능한 화면 크기 가져오기
    QScreen *screen = QApplication::primaryScreen();
    if (!screen) return;
    
    QRect screenGeometry = screen->availableGeometry();
    
    // 전체화면으로 창 크기 설정
    int windowWidth = screenGeometry.width();
    int windowHeight = screenGeometry.height();
    
    // 최소 크기 설정
    setMinimumSize(600, 500);
    
    // 창 크기 설정 (전체화면)
    resize(windowWidth, windowHeight);
    
    // 창을 화면 왼쪽 상단에 위치
    move(screenGeometry.x(), screenGeometry.y());
    
    qDebug() << "Screen geometry:" << screenGeometry;
    qDebug() << "Window size set to:" << windowWidth << "x" << windowHeight;
    qDebug() << "Window positioned at:" << screenGeometry.x() << "," << screenGeometry.y();
}

void MainWindow::setupCamera()
{
    // 사용 가능한 카메라 목록 확인
    const QList<QCameraInfo> availableCameras = QCameraInfo::availableCameras();

    if (availableCameras.isEmpty()) {
        QMessageBox::warning(this, "No Camera", "No camera detected on this system!");
        return;
    }

    // 기본 카메라 사용
    camera = std::make_unique<QCamera>(availableCameras.first());

    // 카메라 출력을 비디오 아이템에 연결
    camera->setViewfinder(videoItem.get());

    // 이미지 캡처 설정
    imageCapture = std::make_unique<QCameraImageCapture>(camera.get());

    // 캡처 이미지 처리 시그널 연결
    connect(imageCapture.get(), &QCameraImageCapture::imageCaptured,
            this, &MainWindow::processCapturedImage);

    // 카메라 에러 처리
    connect(camera.get(), QOverload<QCamera::Error>::of(&QCamera::error),
            this, &MainWindow::displayCameraError);

    // 카메라 상태 변경 시 UI 업데이트
    connect(camera.get(), &QCamera::stateChanged,
            [this](QCamera::State state) {
                if (state == QCamera::ActiveState) {
                    ui->camStartButton->setText("Stop Camera");
                    ui->snapShotButton->setEnabled(true);
                    isCameraRunning = true;
                } else {
                    ui->camStartButton->setText("Start Camera");
                    ui->snapShotButton->setEnabled(false);
                    isCameraRunning = false;
                }
            });

    qDebug() << "Camera initialized:" << availableCameras.first().description();
}

void MainWindow::startCamera()
{
    if (!camera) {
        QMessageBox::warning(this, "Error", "Camera not initialized!");
        return;
    }

    camera->start();

    // 비디오 아이템을 scene 크기에 맞게 조정
    if (videoItem && scene) {
        QRectF sceneRect = scene->sceneRect();
        videoItem->setSize(sceneRect.size());
        videoItem->setPos(0, 0);  // scene 왼쪽 상단에 위치
        ui->camViewer->fitInView(scene->sceneRect(), Qt::KeepAspectRatio);
    }
}

void MainWindow::stopCamera()
{
    if (camera) {
        camera->stop();
    }
}

void MainWindow::on_camStartButton_clicked()
{
    if (!camera) {
        setupCamera();
        if (!camera) return;
    }

    if (isCameraRunning) {
        stopCamera();
    } else {
        startCamera();
    }
}

void MainWindow::on_snapShotButton_clicked()
{
    if (!isNameEntered) {
        QMessageBox::warning(this, "오류", "먼저 이름을 입력해주세요.");
        return;
    }

    if (!imageCapture || !camera) {
        QMessageBox::warning(this, "Error", "Camera not ready!");
        return;
    }

    if (camera->state() != QCamera::ActiveState) {
        QMessageBox::warning(this, "Error", "Camera is not running!");
        return;
    }

    // 즉시 사진 촬영 진행
    ui->snapShotButton->setEnabled(false);
    ui->snapShotButton->setText("업로드 중...");
    ui->statusbar->showMessage(QString("촬영 중... 사용자: %1").arg(currentUserName), 2000);
    
    // 이미지 캡처
    imageCapture->capture();
}

void MainWindow::processCapturedImage(int requestId, const QImage& img)
{
    Q_UNUSED(requestId);

    // 서버로 이미지 전송
    uploadImageToServer(img);
}

void MainWindow::uploadImageToServer(const QImage& image)
{
    // Progress dialog 생성
    QProgressDialog* progressDialog = new QProgressDialog("Uploading image to server...", "Cancel", 0, 100, this);
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->show();

    // 이미지를 JPEG 형식으로 변환
    QByteArray imageData;
    QBuffer buffer(&imageData);
    buffer.open(QIODevice::WriteOnly);
    image.save(&buffer, "JPG", 90); // 90% 품질로 저장

    // HTTP multipart 요청 생성
    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    // 이미지 파트 생성
    QHttpPart imagePart;
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/jpeg"));
    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader,
                        QVariant(QString("form-data; name=\"image\"; filename=\"snapshot_%1.jpg\"")
                                     .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"))));
    imagePart.setBody(imageData);

    // 추가 메타데이터 파트 (옵션)
    QHttpPart metadataPart;
    metadataPart.setHeader(QNetworkRequest::ContentDispositionHeader,
                           QVariant("form-data; name=\"metadata\""));

    QJsonObject metadata;
    metadata["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    metadata["device_id"] = QSysInfo::machineHostName();
    metadata["image_width"] = image.width();
    metadata["image_height"] = image.height();

    QJsonDocument doc(metadata);
    metadataPart.setBody(doc.toJson());

    multiPart->append(imagePart);
    multiPart->append(metadataPart);

    // 요청 생성
    QNetworkRequest request;
    request.setUrl(QUrl(serverUrl + serverEndpoint));
    request.setRawHeader("User-Agent", "Qt Camera Client 1.0");

    // POST 요청 전송
    QNetworkReply* reply = networkManager->post(request, multiPart);
    multiPart->setParent(reply); // reply가 삭제될 때 multiPart도 삭제

    // Progress 업데이트 연결
    connect(reply, &QNetworkReply::uploadProgress,
            [progressDialog](qint64 bytesSent, qint64 bytesTotal) {
                if (bytesTotal > 0) {
                    int progress = static_cast<int>((bytesSent * 100) / bytesTotal);
                    progressDialog->setValue(progress);
                }
            });

    // 취소 버튼 처리
    connect(progressDialog, &QProgressDialog::canceled,
            [reply]() {
                reply->abort();
            });

    // 완료 시 dialog 삭제
    connect(reply, &QNetworkReply::finished,
            [progressDialog]() {
                progressDialog->deleteLater();
            });

    qDebug() << "Uploading image to:" << serverUrl + serverEndpoint;
    qDebug() << "Image size:" << imageData.size() << "bytes";
}

void MainWindow::onUploadFinished(QNetworkReply* reply)
{
    // 버튼 다시 활성화
    ui->snapShotButton->setEnabled(true);
    ui->snapShotButton->setText("Upload Snapshot");

    // 응답 처리
    if (reply->error() == QNetworkReply::NoError) {
        QByteArray response = reply->readAll();

        // JSON 응답 파싱 시도
        QJsonDocument jsonResponse = QJsonDocument::fromJson(response);
        QString message = "Image uploaded successfully!";

        if (!jsonResponse.isNull() && jsonResponse.isObject()) {
            QJsonObject obj = jsonResponse.object();
            if (obj.contains("message")) {
                message = obj["message"].toString();
            }
            if (obj.contains("file_id")) {
                message += QString("\nFile ID: %1").arg(obj["file_id"].toString());
            }
        }

        ui->statusbar->showMessage("Upload successful - Fetching analysis result...", 3000);
        qDebug() << "Server response:" << response;
        
        // 업로드 성공 후 분석 결과 가져오기 (약간의 지연 후)
        QTimer::singleShot(2000, this, &MainWindow::fetchAnalysisResult);
        
    } else {
        QString errorMsg = QString("Upload failed!\nError: %1\n%2")
        .arg(reply->error())
            .arg(reply->errorString());

        // HTTP 상태 코드 확인
        int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if (statusCode) {
            errorMsg += QString("\nHTTP Status: %1").arg(statusCode);
        }

        QMessageBox::critical(this, "Upload Error", errorMsg);
        ui->statusbar->showMessage("Upload failed", 3000);

        qDebug() << "Upload error:" << reply->errorString();
    }

    reply->deleteLater();
}

void MainWindow::onUploadProgress(qint64 bytesSent, qint64 bytesTotal)
{
    if (bytesTotal > 0) {
        int progress = static_cast<int>((bytesSent * 100) / bytesTotal);
        ui->statusbar->showMessage(QString("Uploading... %1%").arg(progress));
    }
}

void MainWindow::displayCameraError()
{
    if (camera) {
        QMessageBox::critical(this, "Camera Error", camera->errorString());
    }
}

void MainWindow::setupUILayout()
{
    // 기존 UI가 레이아웃이 없는 경우, 코드로 추가

    // 메인 레이아웃 생성
    QVBoxLayout *mainLayout = new QVBoxLayout();
    mainLayout->setContentsMargins(15, 15, 15, 15);
    mainLayout->setSpacing(20);  // 간격을 늘려서 버튼과 카메라 분리

    // 카메라 뷰어를 동적 크기 조정 가능하게 설정
    ui->camViewer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->camViewer->setMinimumSize(480, 360);  // 최소 크기를 더 크게 설정 (16:9 비율)
    ui->camViewer->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);  // 최대 크기 제한 제거
    ui->camViewer->setAlignment(Qt::AlignCenter);  // 중앙 정렬
    
    // 카메라 뷰어를 메인 레이아웃에 직접 추가하여 공간을 최대한 활용
    mainLayout->addWidget(ui->camViewer, 1);  // stretch factor를 1로 설정하여 확장 가능

    // 카메라와 버튼 사이에 여백 추가
    mainLayout->addSpacing(15);

    // 버튼 레이아웃 생성 (카메라 화면 밖에 배치)
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(30);  // 버튼 간격 더 증가

    // 버튼들을 중앙에 배치
    buttonLayout->addStretch();  // 왼쪽 여백

    // 버튼 크기를 동적으로 조정 가능하게 설정
    ui->camStartButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    ui->camStartButton->setMinimumSize(180, 50);  // 크기 더 증가
    ui->camStartButton->setMaximumSize(300, 70);  // 최대 크기도 증가
    ui->camStartButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #3498db;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2980b9;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #21618c;"
        "}"
    );
    buttonLayout->addWidget(ui->camStartButton);

    ui->snapShotButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    ui->snapShotButton->setMinimumSize(180, 50);  // 크기 더 증가
    ui->snapShotButton->setMaximumSize(300, 70);  // 최대 크기도 증가
    ui->snapShotButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #27ae60;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2ecc71;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #239653;"
        "}"
        "QPushButton:disabled {"
        "    background-color: #95a5a6;"
        "}"
    );
    buttonLayout->addWidget(ui->snapShotButton);

    buttonLayout->addStretch();  // 오른쪽 여백

    // 버튼 레이아웃을 메인 레이아웃에 추가
    mainLayout->addLayout(buttonLayout, 0);  // stretch factor 0으로 고정

    // centralwidget에 레이아웃 설정
    ui->centralwidget->setLayout(mainLayout);
}

void MainWindow::fetchAnalysisResult()
{
    // rasp.py에서 분석 결과 가져오기 - 별도의 네트워크 매니저 사용
    QNetworkAccessManager* analysisNetworkManager = new QNetworkAccessManager(this);
    
    QNetworkRequest request;
    request.setUrl(QUrl(raspUrl + "/get_analysis"));
    request.setRawHeader("User-Agent", "Qt Camera Client 1.0");
    request.setRawHeader("Accept", "application/json");
    request.setRawHeader("Connection", "close");
    
    QNetworkReply* reply = analysisNetworkManager->get(request);
    
    // 타임아웃 설정
    QTimer::singleShot(10000, reply, &QNetworkReply::abort); // 10초 타임아웃
    
    connect(reply, &QNetworkReply::finished, [this, reply, analysisNetworkManager]() {
        qDebug() << "Network reply finished with error:" << reply->error();
        qDebug() << "HTTP status code:" << reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        qDebug() << "Bytes available:" << reply->bytesAvailable();
        
        // 모든 데이터를 받을 때까지 기다림
        reply->waitForReadyRead(3000);
        
        if (reply->error() == QNetworkReply::NoError) {
            QByteArray response;
            
            // 데이터를 조각별로 읽기
            while (!reply->atEnd()) {
                QByteArray chunk = reply->read(1024);
                response.append(chunk);
                qDebug() << "Read chunk of size:" << chunk.size();
            }
            
            // readAll()로도 한번 더 시도
            QByteArray remaining = reply->readAll();
            response.append(remaining);
            
            qDebug() << "Raw response:" << response;
            qDebug() << "Response length:" << response.length();
            
            if (response.isEmpty()) {
                qDebug() << "Empty response received!";
                ui->statusbar->showMessage("Empty response from server", 3000);
                reply->deleteLater();
                return;
            }
            
            QJsonParseError parseError;
            QJsonDocument jsonDoc = QJsonDocument::fromJson(response, &parseError);
            
            if (parseError.error != QJsonParseError::NoError) {
                qDebug() << "JSON parse error:" << parseError.errorString();
                qDebug() << "Invalid JSON response:" << response;
                ui->statusbar->showMessage("Invalid JSON response", 3000);
                reply->deleteLater();
                return;
            }
            
            if (jsonDoc.isObject()) {
                QJsonObject responseObj = jsonDoc.object();
                qDebug() << "Parsed JSON object:" << responseObj;
                
                if (responseObj["status"].toString() == "success" && 
                    responseObj.contains("analysis_data")) {
                    
                    QJsonObject analysisData = responseObj["analysis_data"].toObject();
                    
                    qDebug() << "Analysis result received:" << analysisData;
                    
                    // 이전 기록 조회 (비교용)
                    DatabaseManager& dbManager = DatabaseManager::instance();
                    QList<AnalysisRecord> userHistory = dbManager.getUserHistory(currentUserName);
                    
                    AnalysisResultDialog *dialog = nullptr;
                    
                    // 이전 기록이 있으면 비교 모드로 표시
                    if (!userHistory.isEmpty()) {
                        QJsonObject previousData = userHistory.first().analysisData;
                        qDebug() << "Found previous record for comparison";
                        
                        // 비교 다이얼로그 생성
                        dialog = new AnalysisResultDialog(analysisData, previousData, currentUserName, this);
                    } else {
                        // 이전 기록이 없으면 일반 모드로 표시
                        qDebug() << "No previous record found, showing normal view";
                        dialog = new AnalysisResultDialog(analysisData, currentUserName, this);
                    }
                    
                    // 현재 분석 결과를 데이터베이스에 저장 (다이얼로그 표시 후)
                    if (dbManager.saveAnalysisResult(currentUserName, analysisData)) {
                        qDebug() << "Analysis result saved to database successfully";
                    } else {
                        qWarning() << "Failed to save analysis result to database";
                    }
                    
                    // 다이얼로그 표시
                    if (dialog) {
                        dialog->exec();
                        dialog->deleteLater();
                    }
                    
                    ui->statusbar->showMessage("Analysis result displayed and saved", 3000);
                    
                } else if (responseObj["status"].toString() == "no_data") {
                    ui->statusbar->showMessage("No analysis data available yet", 3000);
                    qDebug() << "No analysis data available";
                } else {
                    ui->statusbar->showMessage("Failed to get analysis result", 3000);
                    qDebug() << "Failed to get analysis result:" << responseObj;
                }
            } else {
                ui->statusbar->showMessage("Response is not JSON object", 3000);
                qDebug() << "Response is not JSON object:" << jsonDoc;
            }
        } else {
            ui->statusbar->showMessage("Failed to connect to analysis server", 3000);
            qDebug() << "Network error:" << reply->errorString();
            qDebug() << "Error code:" << reply->error();
        }
        
        reply->deleteLater();
        analysisNetworkManager->deleteLater();
    });
    
    qDebug() << "Fetching analysis result from:" << raspUrl + "/get_analysis";
}

void MainWindow::initializeDatabase()
{
    DatabaseManager& dbManager = DatabaseManager::instance();
    if (!dbManager.initializeDatabase()) {
        QMessageBox::critical(this, "데이터베이스 오류", 
            "데이터베이스를 초기화할 수 없습니다.\n애플리케이션을 계속 사용할 수 있지만 분석 결과가 저장되지 않습니다.");
        return;
    }
    
    qDebug() << "Database initialized successfully";
    
    // 통계 정보 로드
    int totalUsers = dbManager.getTotalUsers();
    int totalRecords = dbManager.getTotalRecords();
    QDateTime lastActivity = dbManager.getLastActivity();
    
    QString statsMsg = QString("DB 통계 - 사용자: %1명, 분석기록: %2건")
                      .arg(totalUsers).arg(totalRecords);
    
    if (lastActivity.isValid()) {
        statsMsg += QString(", 마지막 활동: %1").arg(lastActivity.toString("MM-dd hh:mm"));
    }
    
    qDebug() << statsMsg;
}

void MainWindow::setupInitialView()
{
    // 카메라 관련 UI 숨기기
    ui->camViewer->hide();
    ui->camStartButton->hide();
    ui->snapShotButton->hide();
    
    // 중앙 위젯에 이름 입력 단계 UI 추가
    QWidget *nameInputWidget = new QWidget();
    QVBoxLayout *nameLayout = new QVBoxLayout(nameInputWidget);
    nameLayout->setAlignment(Qt::AlignCenter);
    nameLayout->setSpacing(20);
    
    // 환영 메시지
    QLabel *welcomeLabel = new QLabel("피부 개선 디스펜서에 오신 것을 환영합니다!");
    welcomeLabel->setAlignment(Qt::AlignCenter);
    welcomeLabel->setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 20px;");
    nameLayout->addWidget(welcomeLabel);
    
    // 안내 메시지
    QLabel *instructionLabel = new QLabel("피부 분석을 시작하기 전에 이름을 입력해주세요.");
    instructionLabel->setAlignment(Qt::AlignCenter);
    instructionLabel->setStyleSheet("font-size: 14px; color: #34495e; margin: 10px;");
    nameLayout->addWidget(instructionLabel);
    
    // 이름 입력 필드
    QLineEdit *nameLineEdit = new QLineEdit();
    nameLineEdit->setPlaceholderText("이름을 입력하세요 (2-20자)");
    nameLineEdit->setMaxLength(20);
    nameLineEdit->setMaximumWidth(300);
    nameLineEdit->setStyleSheet(
        "QLineEdit {"
        "    padding: 12px;"
        "    border: 2px solid #bdc3c7;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    background-color: white;"
        "}"
        "QLineEdit:focus {"
        "    border-color: #3498db;"
        "    outline: none;"
        "}"
    );
    nameLayout->addWidget(nameLineEdit, 0, Qt::AlignCenter);
    
    // 버튼 레이아웃
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(15);
    
    QPushButton *startButton = new QPushButton("피부 분석 시작");
    startButton->setMinimumSize(150, 45);
    startButton->setStyleSheet(
        "QPushButton {"
        "    padding: 12px 24px;"
        "    background-color: #27ae60;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2ecc71;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #239653;"
        "}"
    );
    
    QPushButton *exitButton = new QPushButton("종료");
    exitButton->setMinimumSize(100, 45);
    exitButton->setStyleSheet(
        "QPushButton {"
        "    padding: 12px 24px;"
        "    background-color: #e74c3c;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "}"
        "QPushButton:hover {"
        "    background-color: #c0392b;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #a93226;"
        "}"
    );
    
    buttonLayout->addStretch();
    buttonLayout->addWidget(startButton);
    buttonLayout->addWidget(exitButton);
    buttonLayout->addStretch();
    
    nameLayout->addLayout(buttonLayout);
    nameLayout->addStretch();
    
    // 현재 중앙 위젯을 nameInputWidget으로 교체
    setCentralWidget(nameInputWidget);
    
    // 연결 설정
    connect(startButton, &QPushButton::clicked, [this, nameLineEdit]() {
        QString name = nameLineEdit->text().trimmed();
        if (name.length() < 2) {
            QMessageBox::warning(this, "입력 오류", "이름은 최소 2자 이상 입력해주세요.");
            nameLineEdit->setFocus();
            return;
        }
        if (name.length() > 20) {
            QMessageBox::warning(this, "입력 오류", "이름은 최대 20자까지 입력 가능합니다.");
            nameLineEdit->setFocus();
            return;
        }
        
        currentUserName = name;
        isNameEntered = true;
        
        qDebug() << "User name entered:" << currentUserName;
        ui->statusbar->showMessage(QString("사용자: %1님, 환영합니다!").arg(currentUserName), 3000);
        
        // 카메라 화면으로 전환
        switchToCameraView();
    });
    
    connect(exitButton, &QPushButton::clicked, this, &QWidget::close);
    
    connect(nameLineEdit, &QLineEdit::returnPressed, [startButton]() {
        startButton->click();
    });
    
    nameLineEdit->setFocus();
}

void MainWindow::switchToCameraView()
{
    // 새로운 중앙 위젯 생성 및 UI 설정
    QWidget *newCentralWidget = new QWidget();
    setCentralWidget(newCentralWidget);
    
    // 새로운 UI 요소들 생성
    QGraphicsView *camViewer = new QGraphicsView(newCentralWidget);
    QPushButton *camStartButton = new QPushButton("카메라 시작", newCentralWidget);
    QPushButton *snapShotButton = new QPushButton("사진 촬영", newCentralWidget);
    
    // UI 포인터 업데이트
    ui->camViewer = camViewer;
    ui->camStartButton = camStartButton;
    ui->snapShotButton = snapShotButton;
    ui->centralwidget = newCentralWidget;
    
    // 레이아웃 설정
    QVBoxLayout *mainLayout = new QVBoxLayout(newCentralWidget);
    mainLayout->setContentsMargins(15, 15, 15, 15);
    mainLayout->setSpacing(20);  // 간격을 늘려서 버튼과 카메라 분리

    // 카메라 뷰어를 동적 크기 조정 가능하게 설정
    camViewer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    camViewer->setMinimumSize(480, 360);  // 최소 크기를 더 크게 설정 (16:9 비율)
    camViewer->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);  // 최대 크기 제한 제거
    camViewer->setAlignment(Qt::AlignCenter);  // 중앙 정렬
    
    // 카메라 뷰어를 메인 레이아웃에 직접 추가하여 공간을 최대한 활용
    mainLayout->addWidget(camViewer, 1);  // stretch factor를 1로 설정하여 확장 가능

    // 카메라와 버튼 사이에 여백 추가
    mainLayout->addSpacing(15);

    // 버튼 레이아웃 (카메라 화면 밖에 배치)
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(30);  // 버튼 간격 더 증가
    buttonLayout->addStretch();
    
    camStartButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    camStartButton->setMinimumSize(180, 50);  // 크기 더 증가
    camStartButton->setMaximumSize(300, 70);  // 최대 크기도 증가
    camStartButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #3498db;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2980b9;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #21618c;"
        "}"
    );
    buttonLayout->addWidget(camStartButton);

    snapShotButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    snapShotButton->setMinimumSize(180, 50);  // 크기 더 증가
    snapShotButton->setMaximumSize(300, 70);  // 최대 크기도 증가
    snapShotButton->setEnabled(false);
    snapShotButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #27ae60;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2ecc71;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #239653;"
        "}"
        "QPushButton:disabled {"
        "    background-color: #95a5a6;"
        "}"
    );
    buttonLayout->addWidget(snapShotButton);
    
    buttonLayout->addStretch();
    mainLayout->addLayout(buttonLayout, 0);
    
    // QGraphicsView 설정
    scene = std::make_unique<QGraphicsScene>(this);
    camViewer->setScene(scene.get());
    
    // 비디오 아이템 생성
    videoItem = std::make_unique<QGraphicsVideoItem>();
    scene->addItem(videoItem.get());
    
    // 시그널 연결
    connect(camStartButton, &QPushButton::clicked, this, &MainWindow::on_camStartButton_clicked);
    connect(snapShotButton, &QPushButton::clicked, this, &MainWindow::on_snapShotButton_clicked);
    
    // 카메라 오버레이 설정 (카메라 초기화보다 먼저)
    setupCameraOverlay();
    
    // 카메라 초기화
    setupCamera();
    
    // 상태바에 현재 사용자 표시
    ui->statusbar->showMessage(QString("현재 사용자: %1").arg(currentUserName), 5000);
}

void MainWindow::showNameInputDialog()
{
    // 이 메소드는 더 이상 사용되지 않지만, 기존 호출을 위해 유지
    // 대신 즉시 사진 촬영 진행
    if (!isNameEntered) {
        QMessageBox::warning(this, "오류", "먼저 이름을 입력해주세요.");
        return;
    }
    
    if (!imageCapture || !camera) {
        QMessageBox::warning(this, "Error", "Camera not ready!");
        return;
    }

    if (camera->state() != QCamera::ActiveState) {
        QMessageBox::warning(this, "Error", "Camera is not running!");
        return;
    }
    
    // 사진 촬영 진행
    ui->snapShotButton->setEnabled(false);
    ui->snapShotButton->setText("업로드 중...");
    ui->statusbar->showMessage(QString("촬영 중... 사용자: %1").arg(currentUserName), 2000);
    
    // 이미지 캡처
    imageCapture->capture();
}

// 오버레이 관련 함수들
void MainWindow::setupCameraOverlay()
{
    if (!scene) {
        return;
    }
    
    // Scene 크기를 카메라 뷰 크기에 맞게 동적 설정
    updateCameraViewSize();
    
    // 얼굴 가이드 원 생성
    faceGuideCircle = new QGraphicsEllipseItem();
    
    // 원의 크기를 뷰 크기에 비례하여 설정
    QSize viewSize = ui->camViewer->size();
    qreal circleSize = qMin(viewSize.width(), viewSize.height()) * 0.8;  // 뷰 크기의 80%로 증가
    
    QRectF sceneRect = scene->sceneRect();
    qreal centerX = sceneRect.center().x();
    qreal centerY = sceneRect.center().y()-25;  // 약간 위쪽으로 이동
    
    faceGuideCircle->setRect(centerX - circleSize/2, centerY - circleSize/2, circleSize, circleSize);
    
    // 원의 스타일 설정
    QPen circlePen(QColor(0, 255, 0, 220), 4);  // 더 진한 초록색, 4픽셀 두께
    circlePen.setStyle(Qt::DashLine);  // 점선 스타일
    faceGuideCircle->setPen(circlePen);
    faceGuideCircle->setBrush(QBrush(Qt::NoBrush));  // 내부는 투명
    
    // Z-order 설정 (비디오 위에 표시)
    faceGuideCircle->setZValue(10);
    
    scene->addItem(faceGuideCircle);
    
    // 안내 텍스트 생성
    guideTextItem = new QGraphicsTextItem("원에 얼굴을 맞춰주세요");
    
    // 텍스트 폰트와 색상 설정
    QFont font("Arial", 18, QFont::Bold);  // 폰트 크기 증가 (16 → 18)
    guideTextItem->setFont(font);
    guideTextItem->setDefaultTextColor(QColor(0, 255, 0));  // 초록색 텍스트 (가독성 향상)
    
    // 텍스트 위치 설정 (원 아래쪽)
    QRectF textRect = guideTextItem->boundingRect();
    qreal textX = centerX - textRect.width() / 2;
    qreal textY = centerY + circleSize/2 + 20;
    guideTextItem->setPos(textX, textY);
    
    // Z-order 설정
    guideTextItem->setZValue(10);
    
    scene->addItem(guideTextItem);
}

void MainWindow::updateOverlayPosition()
{
    if (!faceGuideCircle || !guideTextItem || !scene || !ui->camViewer) {
        return;
    }
    
    // Scene 크기 업데이트
    updateCameraViewSize();
    
    // 현재 뷰 크기에 맞춰 오버레이 크기 및 위치 재조정
    QRectF sceneRect = scene->sceneRect();
    QSize viewSize = ui->camViewer->size();
    qreal circleSize = qMin(viewSize.width(), viewSize.height()) * 0.8;  // 뷰 크기의 80%로 증가
    
    qreal centerX = sceneRect.center().x();
    qreal centerY = sceneRect.center().y()-25;
    
    // 원 위치 및 크기 업데이트
    faceGuideCircle->setRect(centerX - circleSize/2, centerY - circleSize/2, circleSize, circleSize);
    
    // 텍스트 크기를 뷰 크기에 맞게 조정
    QFont font = guideTextItem->font();
    int fontSize = qMax(12, qMin(viewSize.width(), viewSize.height()) / 25);  // 최소 12px, 뷰 크기에 비례
    font.setPointSize(fontSize);
    guideTextItem->setFont(font);
    
    // 텍스트 위치 업데이트
    QRectF textRect = guideTextItem->boundingRect();
    qreal textX = centerX - textRect.width() / 2;
    qreal textY = centerY + circleSize/2 + 20;
    guideTextItem->setPos(textX, textY);
}

void MainWindow::updateCameraViewSize()
{
    if (!scene || !ui->camViewer) {
        return;
    }
    
    // 카메라 뷰 크기에 맞게 Scene 크기 설정
    QSize viewSize = ui->camViewer->size();
    QRectF sceneRect = QRectF(0, 0, viewSize.width(), viewSize.height());
    scene->setSceneRect(sceneRect);
    
    // 배경 업데이트 (기존 배경이 있으면 제거 후 새로 생성)
    QList<QGraphicsItem*> items = scene->items();
    for (QGraphicsItem* item : items) {
        if (item->zValue() == -1) {  // 배경 아이템
            scene->removeItem(item);
            delete item;
            break;
        }
    }
    
    // 새로운 크기의 배경 추가
    QGraphicsRectItem* backgroundRect = scene->addRect(sceneRect, QPen(Qt::NoPen), QBrush(QColor(50, 50, 50)));
    backgroundRect->setZValue(-1);  // 가장 뒤에 배치
    
    // 비디오 아이템 크기도 함께 업데이트
    if (videoItem) {
        videoItem->setSize(sceneRect.size());
        videoItem->setPos(0, 0);
    }
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    
    // 윈도우 크기 변경 시 오버레이 위치와 크기 자동 조정
    if (faceGuideCircle && guideTextItem && scene && ui->camViewer) {
        // 약간의 지연을 두고 업데이트 (레이아웃이 완료된 후)
        QTimer::singleShot(10, this, &MainWindow::updateOverlayPosition);
    }
}

void MainWindow::openNameInputDialog()
{
    qDebug() << "openNameInputDialog called";
    NameInputDialog nameDialog(this);
    if (nameDialog.exec() == QDialog::Accepted) {
        QString name = nameDialog.getUserName();
        if (!name.isEmpty()) {
            currentUserName = name; // currentUserName 업데이트 추가
            isNameEntered = true;
            setWindowTitle(QString("피부 분석 시스템 - %1").arg(name));
            qDebug() << "Name entered:" << name;
            ui->statusbar->showMessage(QString("사용자: %1님, 환영합니다!").arg(currentUserName), 3000);
        }
    }
}