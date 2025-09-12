// mainwindow.h
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <memory>

// Qt Multimedia 헤더들
#include <QtMultimedia/QCamera>
#include <QtMultimedia/QCameraInfo>
#include <QtMultimedia/QCameraImageCapture>
#include <QtMultimediaWidgets/QCameraViewfinder>
#include <QtMultimediaWidgets/QGraphicsVideoItem>
#include <QTimer>
#include <QProcess>
#include <QLabel>
#include <QWidget>
#include "analysisresultdialog.h"
#include "nameinputdialog.h"
#include "databasemanager.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_camStartButton_clicked();
    void on_snapShotButton_clicked();
    void processCapturedImage(int requestId, const QImage& img);
    void displayCameraError();
    void onUploadFinished(QNetworkReply* reply);
    void onUploadProgress(qint64 bytesSent, qint64 bytesTotal);
    void fetchAnalysisResult();
    void showNameInputDialog();
    void openNameInputDialog(); // 새 함수 추가
    void initializeDatabase();
    void updateCameraPreview(); // rpicam 프리뷰 업데이트
    void startGStreamerCamera(); // GStreamer 카메라 시작
    void stopGStreamerCamera();  // GStreamer 카메라 중지
    void captureWithRpicam();    // rpicam-still로 직접 촬영
    void restartPreview();       // 프리뷰 재시작

private:
    Ui::MainWindow *ui;

    std::unique_ptr<QCamera> camera;
    std::unique_ptr<QGraphicsScene> scene;
    std::unique_ptr<QGraphicsVideoItem> videoItem;
    std::unique_ptr<QCameraImageCapture> imageCapture;
    std::unique_ptr<QNetworkAccessManager> networkManager;
    
    // 오버레이 아이템들
    QGraphicsEllipseItem* faceGuideCircle;
    QGraphicsTextItem* guideTextItem;

    // 서버 설정
    QString serverUrl;
    QString serverEndpoint;
    QString raspUrl;
    int serverPort;

    bool isCameraRunning;
    QString currentUserName;
    bool isNameEntered;
    
    // rpicam 기반 라이브 피드
    QTimer* previewTimer;
    QLabel* cameraPreviewLabel;
    bool useRpiCam;
    
    // GStreamer 비디오 스트림
    QProcess* gstreamerProcess;
    QWidget* videoWidget;
    
    void setupCamera();
    void startCamera();
    void stopCamera();
    void uploadImageToServer(const QImage& image);
    void loadServerConfig();
    void setupUILayout();
    void setupInitialView();
    void switchToCameraView();
    void setupCameraOverlay();
    void updateOverlayPosition();
    void updateCameraViewSize();
    void setupWindowSizing();

protected:
    void resizeEvent(QResizeEvent *event) override;
};

#endif // MAINWINDOW_H
