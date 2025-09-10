#include "databasemanager.h"
#include <QDebug>
#include <QDir>
#include <QStandardPaths>
#include <QJsonDocument>
#include <QApplication>

DatabaseManager::DatabaseManager()
{
    // SQLite 드라이버 확인
    if (!QSqlDatabase::isDriverAvailable("QSQLITE")) {
        qCritical() << "SQLite driver not available!";
        return;
    }
}

DatabaseManager::~DatabaseManager()
{
    closeDatabase();
}

DatabaseManager& DatabaseManager::instance()
{
    static DatabaseManager instance;
    return instance;
}

bool DatabaseManager::initializeDatabase()
{
    // 데이터베이스 파일 경로 설정
    QString dataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dataPath);
    QString dbPath = dataPath + "/skin_analysis.db";
    
    qDebug() << "Database path:" << dbPath;
    
    // 데이터베이스 연결 설정
    database = QSqlDatabase::addDatabase("QSQLITE");
    database.setDatabaseName(dbPath);
    
    if (!database.open()) {
        qCritical() << "Failed to open database:" << database.lastError().text();
        return false;
    }
    
    qDebug() << "Database opened successfully";
    
    // 테이블 생성
    if (!createTables()) {
        qCritical() << "Failed to create database tables";
        return false;
    }
    
    return true;
}

void DatabaseManager::closeDatabase()
{
    if (database.isOpen()) {
        database.close();
        qDebug() << "Database closed";
    }
}

bool DatabaseManager::createTables()
{
    QStringList createQueries;
    
    // 분석 결과 테이블
    createQueries << R"(
        CREATE TABLE IF NOT EXISTS analysis_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            capture_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            analysis_data TEXT NOT NULL
        )
    )";
    
    // 인덱스 생성
    createQueries << "CREATE INDEX IF NOT EXISTS idx_records_user ON analysis_records(user_name)";
    createQueries << "CREATE INDEX IF NOT EXISTS idx_records_time ON analysis_records(capture_time)";
    
    for (const QString &query : createQueries) {
        if (!executeQuery(query)) {
            return false;
        }
    }
    
    qDebug() << "Database tables created successfully";
    return true;
}

bool DatabaseManager::executeQuery(const QString &query, const QVariantList &params)
{
    QSqlQuery sqlQuery = prepareQuery(query, params);
    
    if (!sqlQuery.exec()) {
        qCritical() << "Query execution failed:" << sqlQuery.lastError().text();
        qCritical() << "Query:" << query;
        return false;
    }
    
    return true;
}

QSqlQuery DatabaseManager::prepareQuery(const QString &query, const QVariantList &params) const
{
    QSqlQuery sqlQuery(database);
    sqlQuery.prepare(query);
    
    for (int i = 0; i < params.size(); ++i) {
        sqlQuery.bindValue(i, params[i]);
    }
    
    return sqlQuery;
}

bool DatabaseManager::saveAnalysisResult(const QString &userName, const QJsonObject &analysisData)
{
    QString query = R"(
        INSERT INTO analysis_records (user_name, capture_time, analysis_data)
        VALUES (?, CURRENT_TIMESTAMP, ?)
    )";
    
    QJsonDocument doc(analysisData);
    QString jsonString = doc.toJson(QJsonDocument::Compact);
    
    QSqlQuery sqlQuery = prepareQuery(query, {userName, jsonString});
    
    if (sqlQuery.exec()) {
        int recordId = sqlQuery.lastInsertId().toInt();
        qDebug() << "Analysis result saved for:" << userName << "ID:" << recordId;
        return true;
    } else {
        qCritical() << "Failed to save analysis result:" << sqlQuery.lastError().text();
        return false;
    }
}

bool DatabaseManager::updateSessionActivity(int sessionId)
{
    QString query = "UPDATE user_sessions SET last_activity = CURRENT_TIMESTAMP WHERE id = ?";
    return executeQuery(query, {sessionId});
}

bool DatabaseManager::closeUserSession(int sessionId)
{
    QString query = "UPDATE user_sessions SET is_active = 0 WHERE id = ?";
    bool result = executeQuery(query, {sessionId});
    
    if (result && sessionId == currentSessionId) {
        currentSessionId = -1;
    }
    
    return result;
}

UserSession DatabaseManager::getCurrentSession() const
{
    UserSession session;
    session.id = -1;
    
    if (currentSessionId == -1) {
        return session;
    }
    
    QString query = "SELECT id, user_name, session_start, last_activity, is_active FROM user_sessions WHERE id = ?";
    QSqlQuery sqlQuery = prepareQuery(query, {currentSessionId});
    
    if (sqlQuery.exec() && sqlQuery.next()) {
        session.id = sqlQuery.value(0).toInt();
        session.userName = sqlQuery.value(1).toString();
        session.sessionStart = sqlQuery.value(2).toDateTime();
        session.lastActivity = sqlQuery.value(3).toDateTime();
        session.isActive = sqlQuery.value(4).toBool();
    }
    
    return session;
}

QList<UserSession> DatabaseManager::getAllSessions() const
{
    QList<UserSession> sessions;
    QString query = "SELECT id, user_name, session_start, last_activity, is_active FROM user_sessions ORDER BY session_start DESC";
    
    QSqlQuery sqlQuery = prepareQuery(query);
    
    if (sqlQuery.exec()) {
        while (sqlQuery.next()) {
            UserSession session;
            session.id = sqlQuery.value(0).toInt();
            session.userName = sqlQuery.value(1).toString();
            session.sessionStart = sqlQuery.value(2).toDateTime();
            session.lastActivity = sqlQuery.value(3).toDateTime();
            session.isActive = sqlQuery.value(4).toBool();
            sessions.append(session);
        }
    }
    
    return sessions;
}

int DatabaseManager::savePhotoRecord(int sessionId, const QString &fileName, 
                                   const QString &filePath, const QString &metadata)
{
    QString query = R"(
        INSERT INTO photo_records (session_id, file_name, file_path, capture_time, metadata)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
    )";
    
    QSqlQuery sqlQuery = prepareQuery(query, {sessionId, fileName, filePath, metadata});
    
    if (sqlQuery.exec()) {
        int photoId = sqlQuery.lastInsertId().toInt();
        qDebug() << "Photo record saved:" << fileName << "ID:" << photoId;
        
        // 세션 활동 시간 업데이트
        updateSessionActivity(sessionId);
        
        return photoId;
    } else {
        qCritical() << "Failed to save photo record:" << sqlQuery.lastError().text();
        return -1;
    }
}

QList<PhotoRecord> DatabaseManager::getPhotosForSession(int sessionId) const
{
    QList<PhotoRecord> photos;
    QString query = "SELECT id, session_id, file_name, file_path, capture_time, metadata FROM photo_records WHERE session_id = ? ORDER BY capture_time DESC";
    
    QSqlQuery sqlQuery = prepareQuery(query, {sessionId});
    
    if (sqlQuery.exec()) {
        while (sqlQuery.next()) {
            PhotoRecord photo;
            photo.id = sqlQuery.value(0).toInt();
            photo.sessionId = sqlQuery.value(1).toInt();
            photo.fileName = sqlQuery.value(2).toString();
            photo.filePath = sqlQuery.value(3).toString();
            photo.captureTime = sqlQuery.value(4).toDateTime();
            photo.metadata = sqlQuery.value(5).toString();
            photos.append(photo);
        }
    }
    
    return photos;
}

QList<PhotoRecord> DatabaseManager::getAllPhotos() const
{
    QList<PhotoRecord> photos;
    QString query = "SELECT id, session_id, file_name, file_path, capture_time, metadata FROM photo_records ORDER BY capture_time DESC";
    
    QSqlQuery sqlQuery = prepareQuery(query);
    
    if (sqlQuery.exec()) {
        while (sqlQuery.next()) {
            PhotoRecord photo;
            photo.id = sqlQuery.value(0).toInt();
            photo.sessionId = sqlQuery.value(1).toInt();
            photo.fileName = sqlQuery.value(2).toString();
            photo.filePath = sqlQuery.value(3).toString();
            photo.captureTime = sqlQuery.value(4).toDateTime();
            photo.metadata = sqlQuery.value(5).toString();
            photos.append(photo);
        }
    }
    
    return photos;
}

bool DatabaseManager::deletePhotoRecord(int photoId)
{
    QString query = "DELETE FROM photo_records WHERE id = ?";
    return executeQuery(query, {photoId});
}

int DatabaseManager::getTotalUsers() const
{
    QString query = "SELECT COUNT(DISTINCT user_name) FROM user_sessions";
    QSqlQuery sqlQuery = prepareQuery(query);
    
    if (sqlQuery.exec() && sqlQuery.next()) {
        return sqlQuery.value(0).toInt();
    }
    
    return 0;
}

int DatabaseManager::getTotalPhotos() const
{
    QString query = "SELECT COUNT(*) FROM photo_records";
    QSqlQuery sqlQuery = prepareQuery(query);
    
    if (sqlQuery.exec() && sqlQuery.next()) {
        return sqlQuery.value(0).toInt();
    }
    
    return 0;
}

QDateTime DatabaseManager::getLastActivity() const
{
    QString query = "SELECT MAX(last_activity) FROM user_sessions";
    QSqlQuery sqlQuery = prepareQuery(query);
    
    if (sqlQuery.exec() && sqlQuery.next()) {
        return sqlQuery.value(0).toDateTime();
    }
    
    return QDateTime();
}