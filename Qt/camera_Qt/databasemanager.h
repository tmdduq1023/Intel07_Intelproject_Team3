#ifndef DATABASEMANAGER_H
#define DATABASEMANAGER_H

#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlQuery>
#include <QtSql/QSqlError>
#include <QDateTime>
#include <QString>
#include <QVariant>

struct UserSession {
    int id;
    QString userName;
    QDateTime sessionStart;
    QDateTime lastActivity;
    bool isActive;
};

struct PhotoRecord {
    int id;
    int sessionId;
    QString fileName;
    QString filePath;
    QDateTime captureTime;
    QString metadata;
};

class DatabaseManager
{
public:
    static DatabaseManager& instance();
    
    bool initializeDatabase();
    void closeDatabase();
    
    // User Session Management
    int createUserSession(const QString &userName);
    bool updateSessionActivity(int sessionId);
    bool closeUserSession(int sessionId);
    UserSession getCurrentSession() const;
    QList<UserSession> getAllSessions() const;
    
    // Photo Management
    int savePhotoRecord(int sessionId, const QString &fileName, 
                       const QString &filePath, const QString &metadata = QString());
    QList<PhotoRecord> getPhotosForSession(int sessionId) const;
    QList<PhotoRecord> getAllPhotos() const;
    bool deletePhotoRecord(int photoId);
    
    // Statistics
    int getTotalUsers() const;
    int getTotalPhotos() const;
    QDateTime getLastActivity() const;
    
private:
    DatabaseManager();
    ~DatabaseManager();
    
    // Disable copy constructor and assignment operator
    DatabaseManager(const DatabaseManager&) = delete;
    DatabaseManager& operator=(const DatabaseManager&) = delete;
    
    QSqlDatabase database;
    int currentSessionId;
    
    bool createTables();
    bool executeQuery(const QString &query, const QVariantList &params = QVariantList());
    QSqlQuery prepareQuery(const QString &query, const QVariantList &params = QVariantList()) const;
};

#endif // DATABASEMANAGER_H