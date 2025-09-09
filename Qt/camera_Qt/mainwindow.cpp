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
#include <QFileInfo>
#include <QStandardPaths>
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , isCameraRunning(false)
    , currentSessionId(-1)
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

    // 카메라 초기화
    setupCamera();

    // 버튼 텍스트 수정
    ui->camStartButton->setText("Start Camera");
    ui->snapShotButton->setText("Upload Snapshot");
    ui->snapShotButton->setEnabled(false);

    // 데이터베이스 초기화
    initializeDatabase();
    
    // 이름 입력 다이얼로그 표시 (약간의 지연 후)
    QTimer::singleShot(100, this, &MainWindow::showNameInputDialog);
    
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
    serverUrl = settings.value("Server/url", "http://192.168.1.100:8080").toString();
    serverEndpoint = settings.value("Server/endpoint", "/upload").toString();

    // 설정 파일이 없으면 생성
    if (!settings.contains("Server/url")) {
        settings.setValue("Server/url", serverUrl);
        settings.setValue("Server/endpoint", serverEndpoint);
        settings.sync();

        qDebug() << "Created config.ini with default server settings";
    }

    qDebug() << "Server URL:" << serverUrl + serverEndpoint;
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

    // 버튼 비활성화 (중복 전송 방지)
    ui->snapShotButton->setEnabled(false);
    ui->snapShotButton->setText("Uploading...");

    // 이미지 캡처
    imageCapture->capture();
}

void MainWindow::processCapturedImage(int requestId, const QImage& img)
{
    Q_UNUSED(requestId);
    
    // 사진 파일명 생성
    QString fileName = generatePhotoFileName();
    
    // 사진을 로컬에 임시 저장
    QString tempPath = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    QDir().mkpath(tempPath);
    QString filePath = tempPath + "/" + fileName;
    
    if (img.save(filePath, "JPG", 90)) {
        // 데이터베이스에 저장
        savePhotoToDatabase(fileName, filePath);
        qDebug() << "Photo saved locally:" << filePath;
    }

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

        QMessageBox::information(this, "Success", message);
        ui->statusbar->showMessage("Upload successful", 3000);

        qDebug() << "Server response:" << response;
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

void MainWindow::initializeDatabase()
{
    DatabaseManager& dbManager = DatabaseManager::instance();
    if (!dbManager.initializeDatabase()) {
        QMessageBox::critical(this, "데이터베이스 오류", 
            "데이터베이스를 초기화할 수 없습니다.\n애플리케이션을 종료합니다.");
        QApplication::quit();
        return;
    }
    
    qDebug() << "Database initialized successfully";
}

void MainWindow::showNameInputDialog()
{
    NameInputDialog dialog(this);
    
    if (dialog.exec() == QDialog::Accepted) {
        currentUserName = dialog.getUserName();
        
        // 사용자 세션 생성
        DatabaseManager& dbManager = DatabaseManager::instance();
        currentSessionId = dbManager.createUserSession(currentUserName);
        
        if (currentSessionId != -1) {
            updateUIForUser();
            qDebug() << "User session created for:" << currentUserName;
        } else {
            QMessageBox::critical(this, "데이터베이스 오류", 
                "사용자 세션을 생성할 수 없습니다.");
        }
    } else {
        // 사용자가 취소한 경우 애플리케이션 종료
        QApplication::quit();
    }
}

void MainWindow::onNewUserSession()
{
    // 현재 세션 종료
    if (currentSessionId != -1) {
        DatabaseManager::instance().closeUserSession(currentSessionId);
    }
    
    // 카메라 중지
    if (isCameraRunning) {
        stopCamera();
    }
    
    // 새로운 사용자 입력 다이얼로그 표시
    showNameInputDialog();
}

void MainWindow::updateUIForUser()
{
    // 윈도우 제목 업데이트
    setWindowTitle(QString("피부 분석 시스템 - %1").arg(currentUserName));
    
    // 상태바 메시지 업데이트
    ui->statusbar->showMessage(QString("사용자: %1 | Server: %2").arg(currentUserName, serverUrl));
    
    // 새 사용자 버튼을 메뉴에 추가 (메뉴가 있는 경우)
    if (ui->menuCAM) {
        ui->menuCAM->clear();
        QAction *newUserAction = ui->menuCAM->addAction("새 사용자");
        connect(newUserAction, &QAction::triggered, this, &MainWindow::onNewUserSession);
        
        ui->menuCAM->addSeparator();
        
        // 통계 정보 표시 액션
        QAction *statsAction = ui->menuCAM->addAction("통계 정보");
        connect(statsAction, &QAction::triggered, [this]() {
            DatabaseManager& dbManager = DatabaseManager::instance();
            int totalUsers = dbManager.getTotalUsers();
            int totalPhotos = dbManager.getTotalPhotos();
            QDateTime lastActivity = dbManager.getLastActivity();
            
            QString statsMsg = QString(
                "=== 피부 분석 시스템 통계 ===\n\n"
                "총 사용자 수: %1명\n"
                "총 사진 수: %2장\n"
                "마지막 활동: %3\n"
                "현재 사용자: %4"
            ).arg(totalUsers)
             .arg(totalPhotos)
             .arg(lastActivity.isValid() ? lastActivity.toString("yyyy-MM-dd hh:mm:ss") : "없음")
             .arg(currentUserName);
            
            QMessageBox::information(this, "시스템 통계", statsMsg);
        });
    }
}

QString MainWindow::generatePhotoFileName() const
{
    QDateTime now = QDateTime::currentDateTime();
    QString timestamp = now.toString("yyyyMMdd_HHmmss");
    return QString("%1_%2.jpg").arg(currentUserName, timestamp);
}

void MainWindow::savePhotoToDatabase(const QString& fileName, const QString& filePath)
{
    if (currentSessionId == -1) {
        qWarning() << "No active user session for saving photo";
        return;
    }
    
    // 메타데이터 생성
    QJsonObject metadata;
    metadata["user_name"] = currentUserName;
    metadata["session_id"] = currentSessionId;
    metadata["device_id"] = QSysInfo::machineHostName();
    metadata["server_url"] = serverUrl;
    
    QFileInfo fileInfo(filePath);
    metadata["file_size"] = fileInfo.size();
    
    QJsonDocument metaDoc(metadata);
    QString metaString = metaDoc.toJson(QJsonDocument::Compact);
    
    DatabaseManager& dbManager = DatabaseManager::instance();
    int photoId = dbManager.savePhotoRecord(currentSessionId, fileName, filePath, metaString);
    
    if (photoId != -1) {
        qDebug() << "Photo saved to database with ID:" << photoId;
        
        // 사진 저장 성공 메시지를 상태바에 표시
        ui->statusbar->showMessage(QString("사진 저장됨: %1").arg(fileName), 3000);
    } else {
        qWarning() << "Failed to save photo to database";
    }
}
