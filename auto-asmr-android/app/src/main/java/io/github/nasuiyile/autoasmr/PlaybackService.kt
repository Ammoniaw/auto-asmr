package io.github.nasuiyile.autoasmr

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.drawable.Icon
import android.media.MediaMetadata
import android.media.session.MediaSession
import android.media.session.PlaybackState
import android.os.IBinder

/**
 * 前台服务（媒体播放类型）：有声音时让 App 在后台、锁屏时继续跑，通知栏和锁屏上可以暂停 / 继续、切效果 / 原声，
 * 耳机按键也能暂停 / 继续（MediaSession）。什么时候开、关由 [Playback] 决定。
 */
class PlaybackService : Service() {
    private var session: MediaSession? = null
    private var last: String? = null

    override fun onCreate() {
        super.onCreate()
        instance = this
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL, "播放", NotificationManager.IMPORTANCE_LOW).apply {
                description = "正在处理网页声音时的播放控制"
                setShowBadge(false)
            },
        )
        session = MediaSession(this, "auto-asmr").apply {
            setCallback(object : MediaSession.Callback() {
                override fun onPlay() = Playback.resume()
                override fun onPause() = Playback.pause()
                override fun onStop() = Playback.pause()
            })
            isActive = true
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACT_TOGGLE -> Playback.togglePause()
            ACT_DRY -> Settings.setDry(!Settings.params.optBoolean("dry"))
        }
        startForeground(ID, build(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK)
        return START_NOT_STICKY
    }

    /** 状态变了就更新通知和 MediaSession（[Playback] 每秒调一次，没变化时什么都不做）。 */
    fun update() {
        val paused = Playback.pausedByUser
        val dry = Settings.params.optBoolean("dry")
        val title = Playback.title()
        val key = "$paused|$dry|$title|${Playback.sounding()}"
        if (key == last) return
        last = key
        getSystemService(NotificationManager::class.java).notify(ID, build())
    }

    private fun build(): Notification {
        val paused = Playback.pausedByUser
        val dry = Settings.params.optBoolean("dry")
        val title = Playback.title()
        val s = session
        s?.setMetadata(MediaMetadata.Builder().putString(MediaMetadata.METADATA_KEY_TITLE, title).putString(MediaMetadata.METADATA_KEY_ARTIST, if (dry) "原声" else "ASMR 效果").build())
        s?.setPlaybackState(
            PlaybackState.Builder()
                .setActions(PlaybackState.ACTION_PLAY or PlaybackState.ACTION_PAUSE or PlaybackState.ACTION_PLAY_PAUSE or PlaybackState.ACTION_STOP)
                .setState(if (paused) PlaybackState.STATE_PAUSED else PlaybackState.STATE_PLAYING, PlaybackState.PLAYBACK_POSITION_UNKNOWN, 1f)
                .build(),
        )
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP), PendingIntent.FLAG_IMMUTABLE)
        fun act(action: String, code: Int) = PendingIntent.getService(this, code, Intent(this, PlaybackService::class.java).setAction(action), PendingIntent.FLAG_IMMUTABLE)
        val icon = Icon.createWithResource(this, R.drawable.ic_stat_asmr)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_asmr)
            .setContentTitle(title)
            .setContentText(if (paused) "已暂停" else if (dry) "原声（没有效果）" else "ASMR 效果")
            .setContentIntent(open)
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_TRANSPORT)
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .addAction(Notification.Action.Builder(icon, if (paused) "继续" else "暂停", act(ACT_TOGGLE, 1)).build())
            .addAction(Notification.Action.Builder(icon, if (dry) "切到效果" else "切到原声", act(ACT_DRY, 2)).build())
            .setStyle(Notification.MediaStyle().setMediaSession(s?.sessionToken).setShowActionsInCompactView(0, 1))
            .build()
    }

    override fun onDestroy() {
        instance = null
        session?.release()
        session = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        const val CHANNEL = "playback"
        const val ID = 1
        const val ACT_TOGGLE = "io.github.nasuiyile.autoasmr.TOGGLE"
        const val ACT_DRY = "io.github.nasuiyile.autoasmr.DRY"
        @Volatile var instance: PlaybackService? = null
    }
}
