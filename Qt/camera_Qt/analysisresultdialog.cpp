#include "analysisresultdialog.h"
#include <QApplication>
#include <QFont>
#include <QFrame>

AnalysisResultDialog::AnalysisResultDialog(QJsonObject analysisData, QString userName, QWidget *parent)
    : QDialog(parent), analysisData(analysisData), userName(userName), isComparison(false)
{
    setWindowTitle(QString("피부 분석 결과 - %1").arg(userName.isEmpty() ? "사용자" : userName));
    setModal(true);
    setMinimumSize(480, 400);
    resize(600, 550);
    
    setupUI();
    displayAnalysisData(analysisData);
}

AnalysisResultDialog::AnalysisResultDialog(QJsonObject currentData, QJsonObject previousData, QString userName, QWidget *parent)
    : QDialog(parent), analysisData(currentData), previousData(previousData), userName(userName), isComparison(true)
{
    setWindowTitle(QString("피부 분석 비교 - %1").arg(userName.isEmpty() ? "사용자" : userName));
    setModal(true);
    setMinimumSize(550, 450);
    resize(700, 600);
    
    setupUI();
    displayComparisonData(currentData, previousData);
}

void AnalysisResultDialog::setupUI()
{
    mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(8);
    mainLayout->setContentsMargins(15, 15, 15, 15);
    
    // 제목 라벨
    QString titleText = isComparison ? "피부 분석 비교" : "피부 분석 결과";
    QLabel *titleLabel = new QLabel(titleText);
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(14);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    titleLabel->setAlignment(Qt::AlignCenter);
    titleLabel->setStyleSheet("color: #2c3e50; margin: 8px 0px;");
    mainLayout->addWidget(titleLabel);
    
    // 사용자 이름 라벨 (있는 경우)
    if (!userName.isEmpty()) {
        QLabel *userLabel = new QLabel(QString("사용자: %1").arg(userName));
        QFont userFont = userLabel->font();
        userFont.setPointSize(10);
        userLabel->setFont(userFont);
        userLabel->setAlignment(Qt::AlignCenter);
        userLabel->setStyleSheet("color: #7f8c8d; margin-bottom: 12px;");
        mainLayout->addWidget(userLabel);
    }
    
    // 비교 모드일 때 안내 라벨
    if (isComparison) {
        QLabel *infoLabel = new QLabel("현재 ↔ 이전 (변화량)");
        QFont infoFont = infoLabel->font();
        infoFont.setPointSize(9);
        infoLabel->setFont(infoFont);
        infoLabel->setAlignment(Qt::AlignCenter);
        infoLabel->setStyleSheet("color: #34495e; background-color: #ecf0f1; padding: 4px; border-radius: 3px; margin-bottom: 10px;");
        mainLayout->addWidget(infoLabel);
    }
    
    // 스크롤 영역 설정
    scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(true);
    scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    
    contentWidget = new QWidget();
    scrollArea->setWidget(contentWidget);
    
    mainLayout->addWidget(scrollArea);
    
    // 닫기 버튼
    closeButton = new QPushButton("닫기");
    closeButton->setMinimumHeight(32);
    closeButton->setStyleSheet(
        "QPushButton {"
        "  background-color: #3498db;"
        "  border: none;"
        "  color: white;"
        "  padding: 6px 12px;"
        "  border-radius: 4px;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2980b9;"
        "}"
        "QPushButton:pressed {"
        "  background-color: #21618c;"
        "}"
    );
    
    connect(closeButton, &QPushButton::clicked, this, &QDialog::accept);
    mainLayout->addWidget(closeButton);
}

void AnalysisResultDialog::displayAnalysisData(const QJsonObject &data)
{
    QVBoxLayout *contentLayout = new QVBoxLayout(contentWidget);
    contentLayout->setSpacing(15);
    contentLayout->setContentsMargins(10, 10, 10, 10);
    
    // 각 영역별로 위젯 생성
    QStringList regions = {"forehead", "l_check", "r_check", "chin", "lib"};
    QStringList regionNames = {"이마", "왼쪽 볼", "오른쪽 볼", "턱", "입술"};
    
    for (int i = 0; i < regions.size(); ++i) {
        if (data.contains(regions[i]) && data[regions[i]].isObject()) {
            QWidget *regionWidget = createRegionWidget(regionNames[i], data[regions[i]].toObject());
            contentLayout->addWidget(regionWidget);
        }
    }
    
    contentLayout->addStretch();
}

QWidget* AnalysisResultDialog::createRegionWidget(const QString &regionName, const QJsonObject &regionData)
{
    QGroupBox *groupBox = new QGroupBox(regionName);
    groupBox->setStyleSheet(
        "QGroupBox {"
        "  font-weight: bold;"
        "  font-size: 12px;"
        "  color: #2c3e50;"
        "  border: 2px solid #bdc3c7;"
        "  border-radius: 6px;"
        "  margin-top: 8px;"
        "  padding-top: 8px;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 8px;"
        "  padding: 0 4px 0 4px;"
        "}"
    );
    
    QGridLayout *gridLayout = new QGridLayout(groupBox);
    gridLayout->setSpacing(8);
    gridLayout->setContentsMargins(10, 15, 10, 10);
    
    int row = 0;
    int col = 0;
    const int maxCols = 2; // 한 줄에 최대 2개 메트릭
    
    // 메트릭들을 그리드로 배치
    for (auto it = regionData.begin(); it != regionData.end(); ++it) {
        if (it.value().isDouble()) {
            int value = static_cast<int>(it.value().toDouble());
            QWidget *metricWidget = createMetricWidget(it.key(), value);
            gridLayout->addWidget(metricWidget, row, col);
            
            col++;
            if (col >= maxCols) {
                col = 0;
                row++;
            }
        }
    }
    
    return groupBox;
}

QWidget* AnalysisResultDialog::createMetricWidget(const QString &metricName, int value)
{
    QWidget *widget = new QWidget();
    widget->setMinimumHeight(65);
    widget->setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 8px;");
    
    QVBoxLayout *layout = new QVBoxLayout(widget);
    layout->setSpacing(6);
    layout->setContentsMargins(10, 8, 10, 8);
    
    // 메트릭 이름
    QLabel *nameLabel = new QLabel(getMetricDescription(metricName));
    QFont nameFont = nameLabel->font();
    nameFont.setPointSize(10);
    nameFont.setBold(true);
    nameLabel->setFont(nameFont);
    nameLabel->setStyleSheet("color: #2c3e50;");
    layout->addWidget(nameLabel);
    
    // 진행률 바
    QProgressBar *progressBar = new QProgressBar();
    progressBar->setRange(0, 100);
    progressBar->setValue(value);
    progressBar->setTextVisible(false);
    progressBar->setFixedHeight(12);
    
    QColor barColor = getScoreColor(value);
    progressBar->setStyleSheet(QString(
        "QProgressBar {"
        "  border: 1px solid #bdc3c7;"
        "  border-radius: 7px;"
        "  background-color: #ecf0f1;"
        "}"
        "QProgressBar::chunk {"
        "  background-color: %1;"
        "  border-radius: 6px;"
        "}"
    ).arg(barColor.name()));
    
    layout->addWidget(progressBar);
    
    // 점수와 설명
    QHBoxLayout *scoreLayout = new QHBoxLayout();
    
    QLabel *scoreLabel = new QLabel(QString("%1").arg(value));
    QFont scoreFont = scoreLabel->font();
    scoreFont.setPointSize(12);
    scoreFont.setBold(true);
    scoreLabel->setFont(scoreFont);
    scoreLabel->setStyleSheet(QString("color: %1;").arg(barColor.name()));
    
    QLabel *descLabel = new QLabel(getScoreDescription(value));
    QFont descFont = descLabel->font();
    descFont.setPointSize(9);
    descLabel->setFont(descFont);
    descLabel->setStyleSheet("color: #7f8c8d;");
    
    scoreLayout->addWidget(scoreLabel);
    scoreLayout->addStretch();
    scoreLayout->addWidget(descLabel);
    
    layout->addLayout(scoreLayout);
    
    return widget;
}

QString AnalysisResultDialog::getMetricDescription(const QString &metricName)
{
    if (metricName == "moisture") return "수분";
    if (metricName == "elasticity") return "탄력";
    if (metricName == "pigmentation") return "색소침착";
    if (metricName == "pore") return "모공";
    return metricName;
}

QString AnalysisResultDialog::getScoreDescription(int score)
{
    if (score >= 80) return "매우 좋음";
    if (score >= 60) return "좋음";
    if (score >= 40) return "보통";
    if (score >= 20) return "주의필요";
    return "관리필요";
}

QColor AnalysisResultDialog::getScoreColor(int score)
{
    if (score >= 80) return QColor("#27ae60"); // 초록
    if (score >= 60) return QColor("#2ecc71"); // 연한 초록
    if (score >= 40) return QColor("#f39c12"); // 주황
    if (score >= 20) return QColor("#e67e22"); // 진한 주황
    return QColor("#e74c3c"); // 빨강
}

void AnalysisResultDialog::displayComparisonData(const QJsonObject &currentData, const QJsonObject &previousData)
{
    QVBoxLayout *contentLayout = new QVBoxLayout(contentWidget);
    contentLayout->setSpacing(12);
    contentLayout->setContentsMargins(8, 8, 8, 8);
    
    // 각 영역별로 비교 위젯 생성
    QStringList regions = {"forehead", "l_check", "r_check", "chin", "lib"};
    QStringList regionNames = {"이마", "왼쪽 볼", "오른쪽 볼", "턱", "입술"};
    
    for (int i = 0; i < regions.size(); ++i) {
        if (currentData.contains(regions[i]) && currentData[regions[i]].isObject()) {
            QJsonObject currentRegion = currentData[regions[i]].toObject();
            QJsonObject previousRegion = previousData.contains(regions[i]) ? 
                previousData[regions[i]].toObject() : QJsonObject();
                
            QWidget *regionWidget = createComparisonRegionWidget(regionNames[i], currentRegion, previousRegion);
            contentLayout->addWidget(regionWidget);
        }
    }
    
    contentLayout->addStretch();
}

QWidget* AnalysisResultDialog::createComparisonRegionWidget(const QString &regionName, const QJsonObject &currentData, const QJsonObject &previousData)
{
    QGroupBox *groupBox = new QGroupBox(regionName);
    groupBox->setStyleSheet(
        "QGroupBox {"
        "  font-weight: bold;"
        "  font-size: 12px;"
        "  color: #2c3e50;"
        "  border: 2px solid #bdc3c7;"
        "  border-radius: 6px;"
        "  margin-top: 8px;"
        "  padding-top: 8px;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 8px;"
        "  padding: 0 4px 0 4px;"
        "}"
    );
    
    QGridLayout *gridLayout = new QGridLayout(groupBox);
    gridLayout->setSpacing(8);
    gridLayout->setContentsMargins(10, 15, 10, 10);
    
    int row = 0;
    int col = 0;
    const int maxCols = 2; // 한 줄에 최대 2개 메트릭
    
    // 메트릭들을 그리드로 배치
    for (auto it = currentData.begin(); it != currentData.end(); ++it) {
        if (it.value().isDouble()) {
            int currentValue = static_cast<int>(it.value().toDouble());
            int previousValue = previousData.contains(it.key()) ? 
                static_cast<int>(previousData[it.key()].toDouble()) : -1;
                
            QWidget *metricWidget = createComparisonMetricWidget(it.key(), currentValue, previousValue);
            gridLayout->addWidget(metricWidget, row, col);
            
            col++;
            if (col >= maxCols) {
                col = 0;
                row++;
            }
        }
    }
    
    return groupBox;
}

QWidget* AnalysisResultDialog::createComparisonMetricWidget(const QString &metricName, int currentValue, int previousValue)
{
    QWidget *widget = new QWidget();
    widget->setMinimumHeight(65);
    widget->setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 8px;");
    
    QVBoxLayout *layout = new QVBoxLayout(widget);
    layout->setSpacing(6);
    layout->setContentsMargins(10, 8, 10, 8);
    
    // 메트릭 이름
    QLabel *nameLabel = new QLabel(getMetricDescription(metricName));
    QFont nameFont = nameLabel->font();
    nameFont.setPointSize(10);
    nameFont.setBold(true);
    nameLabel->setFont(nameFont);
    nameLabel->setStyleSheet("color: #2c3e50;");
    layout->addWidget(nameLabel);
    
    // 현재 값 진행률 바
    QProgressBar *currentBar = new QProgressBar();
    currentBar->setRange(0, 100);
    currentBar->setValue(currentValue);
    currentBar->setTextVisible(false);
    currentBar->setFixedHeight(12);
    
    QColor currentColor = getScoreColor(currentValue);
    currentBar->setStyleSheet(QString(
        "QProgressBar {"
        "  border: 1px solid #bdc3c7;"
        "  border-radius: 6px;"
        "  background-color: #ecf0f1;"
        "}"
        "QProgressBar::chunk {"
        "  background-color: %1;"
        "  border-radius: 5px;"
        "}"
    ).arg(currentColor.name()));
    
    layout->addWidget(currentBar);
    
    // 값과 비교 정보
    QHBoxLayout *infoLayout = new QHBoxLayout();
    
    // 현재 값
    QLabel *currentLabel = new QLabel(QString("%1").arg(currentValue));
    QFont currentFont = currentLabel->font();
    currentFont.setPointSize(12);
    currentFont.setBold(true);
    currentLabel->setFont(currentFont);
    currentLabel->setStyleSheet(QString("color: %1;").arg(currentColor.name()));
    
    infoLayout->addWidget(currentLabel);
    
    // 비교 정보 (이전 값이 있을 경우)
    if (previousValue >= 0) {
        int change = currentValue - previousValue;
        
        QLabel *changeLabel = new QLabel();
        if (change > 0) {
            changeLabel->setText(QString("▲%1").arg(change));
            changeLabel->setStyleSheet("color: #27ae60; font-weight: bold; font-size: 10px;");
        } else if (change < 0) {
            changeLabel->setText(QString("▼%1").arg(-change));
            changeLabel->setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 10px;");
        } else {
            changeLabel->setText("→ 0");
            changeLabel->setStyleSheet("color: #95a5a6; font-weight: bold; font-size: 10px;");
        }
        
        infoLayout->addWidget(changeLabel);
        
        // 이전 값
        QLabel *previousLabel = new QLabel(QString("(%1)").arg(previousValue));
        QFont prevFont = previousLabel->font();
        prevFont.setPointSize(9);
        previousLabel->setFont(prevFont);
        previousLabel->setStyleSheet("color: #7f8c8d;");
        
        infoLayout->addWidget(previousLabel);
    }
    
    infoLayout->addStretch();
    
    // 설명
    QLabel *descLabel = new QLabel(getScoreDescription(currentValue));
    QFont descFont = descLabel->font();
    descFont.setPointSize(9);
    descLabel->setFont(descFont);
    descLabel->setStyleSheet("color: #7f8c8d;");
    
    infoLayout->addWidget(descLabel);
    
    layout->addLayout(infoLayout);
    
    return widget;
}

QString AnalysisResultDialog::getChangeDescription(int change)
{
    if (change > 10) return "크게 개선";
    if (change > 5) return "개선";
    if (change > 0) return "약간 개선";
    if (change == 0) return "변화없음";
    if (change > -5) return "약간 악화";
    if (change > -10) return "악화";
    return "크게 악화";
}

QColor AnalysisResultDialog::getChangeColor(int change)
{
    if (change > 5) return QColor("#27ae60");      // 초록 (개선)
    if (change > 0) return QColor("#2ecc71");      // 연한 초록 (약간 개선)  
    if (change == 0) return QColor("#95a5a6");     // 회색 (변화없음)
    if (change > -5) return QColor("#f39c12");     // 주황 (약간 악화)
    return QColor("#e74c3c");                      // 빨강 (악화)
}