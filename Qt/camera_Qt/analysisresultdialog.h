#ifndef ANALYSISRESULTDIALOG_H
#define ANALYSISRESULTDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QJsonObject>
#include <QScrollArea>

class AnalysisResultDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AnalysisResultDialog(QJsonObject analysisData, QString userName = QString(), QWidget *parent = nullptr);
    explicit AnalysisResultDialog(QJsonObject currentData, QJsonObject previousData, QString userName, QWidget *parent = nullptr);

private:
    void setupUI();
    void displayAnalysisData(const QJsonObject &data);
    void displayComparisonData(const QJsonObject &currentData, const QJsonObject &previousData);
    QWidget* createRegionWidget(const QString &regionName, const QJsonObject &regionData);
    QWidget* createComparisonRegionWidget(const QString &regionName, const QJsonObject &currentData, const QJsonObject &previousData);
    QWidget* createMetricWidget(const QString &metricName, int value);
    QWidget* createComparisonMetricWidget(const QString &metricName, int currentValue, int previousValue);
    QString getMetricDescription(const QString &metricName);
    QString getScoreDescription(int score);
    QColor getScoreColor(int score);
    QString getChangeDescription(int change);
    QColor getChangeColor(int change);
    
    QJsonObject analysisData;
    QJsonObject previousData;
    QString userName;
    bool isComparison;
    QScrollArea *scrollArea;
    QWidget *contentWidget;
    QVBoxLayout *mainLayout;
    QPushButton *closeButton;
};

#endif // ANALYSISRESULTDIALOG_H