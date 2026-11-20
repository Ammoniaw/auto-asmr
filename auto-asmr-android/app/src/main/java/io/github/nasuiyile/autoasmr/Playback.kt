package io.github.nasuiyile.autoasmr

import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.webkit.WebViewCompat

/**
 * 播放的总控：每秒看一次（不依赖界面，切到后台也照常跑）。
 *
 * - **前台服务**（[PlaybackService]）：有声音（网页在出声或雨声在响）就开，通知栏可以暂停 / 继续、切效果 / 原声；
 *   没声音 30 秒后关掉。用户从通知栏暂停的，10 分钟内都留着，方便接着放。前台服务让 App 在后台、锁屏时不被冻结。
 * - **暂停 / 继续**：让正在出声的标签页里所有框架的媒体和 Web Audio 暂停 / 继续（发消息给注入的脚本），雨声也一起停。
 * - **截取自检**：网页在出声、却 4 秒都没有被截到的 AAudio 流，就装上备用的 inline hook，并让这个标签页静音约 6 秒
 *   再恢复（Chromium 关掉旧流、开一条新流，新流就截得到）；装了还截不到，就提示这个 WebView 不支持。
 */
object Playback {
    private const val TAG = "AsmrPlayback"
    private const val KEEP_MS = 30_000L
    private const val KEEP_PAUSED_MS = 10 * 60_000L

    var browser: Browser? = null
    private lateinit var app: Context
    private val main = Handler(Looper.getMainLooper())

    /** 用户从通知栏 / 耳机按键暂停了。 */
    var pausedByUser by mutableStateOf(false)
        private set
    /** 截取有问题时给用户看的提示；没问题时为 null。 */
    var hookProblem by mutableStateOf<String?>(null)
        private set
    var serviceRunning by mutableStateOf(false)
        private set

    private var lastSound = 0L
    private var pausedAt = 0L
    private var noHookSec = 0
    private var triedInline = false

    fun init(context: Context) {
        app = context.applicationContext
        main.post(tick)
    }

    private val tick = object : Runnable {
        override fun run() {
            runCatching { check() }.onFailure { Log.w(TAG, "check", it) }
            main.postDelayed(this, 1000)
        }
    }

    /** 此刻有没有声音（网页在出声或雨声在响）。 */
    fun sounding(): Boolean = browser?.audibleTab != null || NativeHook.rainPlaying()

    fun title(): String {
        val t = browser?.audibleTab ?: (browser?.pinned ?: browser?.current)
        return t?.title?.ifEmpty { t.url }?.takeIf { it.isNotEmpty() } ?: "auto-asmr"
    }

    private fun check() {
        val now = SystemClock.elapsedRealtime()
        val sounding = sounding()
        if (sounding) lastSound = now
        val keep = sounding || now - lastSound < KEEP_MS || (pausedByUser && now - pausedAt < KEEP_PAUSED_MS)
        if (keep && !serviceRunning) startService() else if (!keep && serviceRunning) stopService()
        PlaybackService.instance?.update()
        selfCheck()
    }

    private fun startService() {
        try {
            app.startForegroundService(Intent(app, PlaybackService::class.java))
            serviceRunning = true
        } catch (e: Exception) {
            // Android 12 起不能从后台启动前台服务：声音一般是在前台开始的，那时已经开了，这里只是兜底
            Log.w(TAG, "前台服务启动失败：$e")
        }
    }

    private fun stopService() {
        app.stopService(Intent(app, PlaybackService::class.java))
        serviceRunning = false
    }

    fun pause() {
        val b = browser ?: return
        pausedByUser = true
        pausedAt = SystemClock.elapsedRealtime()
        b.sendToAudible("pause")
        Settings.resyncRain()
        PlaybackService.instance?.update()
    }

    fun resume() {
        val b = browser ?: return
        pausedByUser = false
        lastSound = SystemClock.elapsedRealtime()
        b.sendToAudible("play")
        Settings.resyncRain()
        PlaybackService.instance?.update()
    }

    fun togglePause() = if (pausedByUser) resume() else pause()

    private fun selfCheck() {
        val b = browser ?: return
        val st = NativeHook.hookState()
        if (st[2] > 0) {
            noHookSec = 0
            hookProblem = null
            return
        }
        if (b.audibleTab == null) {
            noHookSec = 0
            return
        }
        noHookSec++
        if (noHookSec != 4) return
        val wv = WebViewCompat.getCurrentWebViewPackage(app)?.let { "${it.packageName} ${it.versionName}" } ?: "未知"
        if (!triedInline && st[1] == 0) {
            triedInline = true
            val fails = NativeHook.installInline()
            Log.w(TAG, "网页在出声但没截到声音（PLT hook 截到的调用方 ${st[0]} 个），改用 inline hook：失败 $fails 个；WebView $wv")
            hookProblem = "没有截获到网页的声音，正在改用备用方式：声音会停几秒再恢复。"
            b.audibleTab?.let { b.reopenAudio(it) }
            noHookSec = -10 // 静音 6 秒、重新开流的这段时间不算
        } else {
            Log.w(TAG, "备用方式也没截到声音；WebView $wv")
            hookProblem = "截不到这个网页的声音（WebView $wv）。声音照常播放，但没有 ASMR 效果。"
        }
    }
}
