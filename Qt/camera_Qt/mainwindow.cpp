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

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , isCameraRunning(false)
{
    ui->setupUi(this);

    // 레이아웃 설정 (UI 파일에 레이아웃이 없는 경우)
    setupUILayout();

    // QGraphicsView 설정
    scene = std::make_unique<QGraphicsScene>(this);
    ui->camViewer->setScene(scene.get());

    // 비디오 아이템 생성
    videoItem = std::make_unique<QGraphicsVideoItem>();
    scene->addItem(videoItem.get());

    // 네트워크 매니저 초기화
    networkManager = std::make_unique<QNetworkAccessManager>(this);
    connect(networkManager.get(), &QNetworkAccessManager::finished,
            this, &MainWindow::onUploadFinished);

    // 서버 설정 로드
    loadServerConfig();

    // 데이터베이스 초기화
    initializeDatabase();
    
    // 카메라 초기화
    setupCamera();

    // 버튼 텍스트 수정
    ui->camStartButton->setText("Start Camera");
    ui->snapShotButton->setText("Upload Snapshot");
    ui->snapShotButton->setEnabled(false);

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

    // 비디오 아이템을 뷰에 맞게 조정
    videoItem->setSize(ui->camViewer->size());
    ui->camViewer->fitInView(videoItem.get(), Qt::KeepAspectRatio);
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
    if (!imageCapture || !camera) {
        QMessageBox::warning(this, "Error", "Camera not ready!");
        return;
    }

    if (camera->state() != QCamera::ActiveState) {
        QMessageBox::warning(this, "Error", "Camera is not running!");
        return;
    }

    // 이름 입력 다이얼로그 표시
    showNameInputDialog();
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
    mainLayout->setContentsMargins(5, 5, 5, 5);
    mainLayout->setSpacing(5);

    // 카메라 뷰어 추가 (확장 가능하게)
    ui->camViewer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    mainLayout->addWidget(ui->camViewer, 1);  // stretch factor 1

    // 버튼 레이아웃 생성
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(10);

    // 버튼들을 중앙에 배치
    buttonLayout->addStretch();  // 왼쪽 여백

    // 버튼 크기 정책 설정
    ui->camStartButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    ui->camStartButton->setMinimumSize(120, 35);
    ui->camStartButton->setMaximumSize(200, 50);
    buttonLayout->addWidget(ui->camStartButton);

    ui->snapShotButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    ui->snapShotButton->setMinimumSize(120, 35);
    ui->snapShotButton->setMaximumSize(200, 50);
    buttonLayout->addWidget(ui->snapShotButton);

    buttonLayout->addStretch();  // 오른쪽 여백

    // 버튼 레이아웃을 메인 레이아웃에 추가
    mainLayout->addLayout(buttonLayout);

    // centralwidget에 레이아웃 설정
    ui->centralwidget->setLayout(mainLayout);

    // 윈도우 최소 크기 설정
    setMinimumSize(640, 480);
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

void MainWindow::showNameInputDialog()
{
    NameInputDialog dialog(this);
    
    if (dialog.exec() == QDialog::Accepted) {
        currentUserName = dialog.getUserName();
        
        qDebug() << "User name entered:" << currentUserName;
        
        // 사용자 이름이 입력되면 사진 촬영 진행
        ui->snapShotButton->setEnabled(false);
        ui->snapShotButton->setText("Uploading...");
        ui->statusbar->showMessage(QString("촬영 중... 사용자: %1").arg(currentUserName), 2000);
        
        // 이미지 캡처
        imageCapture->capture();
        
    } else {
        // 사용자가 취소한 경우
        ui->statusbar->showMessage("촬영이 취소되었습니다.", 2000);
    }
}

