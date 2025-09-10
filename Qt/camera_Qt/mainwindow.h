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
    void initializeDatabase();

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
