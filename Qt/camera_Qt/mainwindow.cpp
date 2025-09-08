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

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , isCameraRunning(false)
{
    ui->setupUi(this);

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
