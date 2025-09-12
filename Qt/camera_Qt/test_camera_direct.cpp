#include <QApplication>
#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <QPixmap>
#include <QDebug>
#include <QProcess>
#include <QTemporaryFile>

class DirectCameraTest : public QWidget
{
    Q_OBJECT

public:
    DirectCameraTest(QWidget *parent = nullptr) : QWidget(parent)
    {
        setWindowTitle("Direct Camera Test - rpicam");
        resize(800, 600);
        
        label = new QLabel("카메라 초기화 중...", this);
        label->setAlignment(Qt::AlignCenter);
        label->setStyleSheet("border: 1px solid black;");
        
        QVBoxLayout *layout = new QVBoxLayout(this);
        layout->addWidget(label);
        
        timer = new QTimer(this);
        connect(timer, &QTimer::timeout, this, &DirectCameraTest::captureFrame);
        
        // 1초마다 프레임 캡처 (테스트용)
        timer->start(1000);
    }

private slots:
    void captureFrame()
    {
        // rpicam으로 직접 이미지 캡처
        QProcess process;
        QString tempFile = "/tmp/camera_frame.jpg";
        
        QStringList args;
        args << "-t" << "1" << "-o" << tempFile 
             << "--width" << "640" << "--height" << "480" 
             << "--nopreview";
        
        qDebug() << "Executing: rpicam-still" << args.join(" ");
        process.start("rpicam-still", args);
        
        if (process.waitForFinished(5000)) {
            if (process.exitCode() == 0) {
                // 캡처된 이미지를 로드하여 표시
                QPixmap pixmap(tempFile);
                if (!pixmap.isNull()) {
                    label->setPixmap(pixmap.scaled(label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
                    qDebug() << "Frame updated successfully";
                } else {
                    label->setText("이미지 로드 실패");
                    qDebug() << "Failed to load captured image";
                }
                
                // 임시 파일 삭제
                QFile::remove(tempFile);
            } else {
                QString error = process.readAllStandardError();
                label->setText(QString("rpicam-still 실행 실패: %1").arg(error));
                qDebug() << "rpicam-still failed:" << error;
            }
        } else {
            label->setText("rpicam-still 타임아웃");
            qDebug() << "rpicam-still timeout";
        }
    }

private:
    QLabel *label;
    QTimer *timer;
};

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    DirectCameraTest window;
    window.show();
    
    return app.exec();
}

#include "test_camera_direct.moc"