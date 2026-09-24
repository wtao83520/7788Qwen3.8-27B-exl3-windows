// ===========================================================================
// pm2 配置：让推理服务常驻（开机自启 + 崩溃自动拉起）
// ---------------------------------------------------------------------------
// 用法（在项目根目录）：
//     pm2 start ecosystem.config.js       # 启动并托管
//     pm2 save                            # 记住当前进程列表（开机自启要用）
//     pm2 logs qwen38-server              # 看日志
//     pm2 stop qwen38-server              # 停止（停止后不会自动重启）
//     pm2 restart qwen38-server           # 重启
//     pm2 delete qwen38-server            # 从 pm2 列表里移除
//
// 开机自启：pm2-startup install （已装 pm2-windows-startup），然后 pm2 save。
//
// ⚠️ 托管之后，启停必须走 pm2（或控制面板，它已经能识别 pm2 并转发）。
//    不要再用 taskkill 或 /admin/shutdown 去“关”它：进程一退出，pm2 会立刻
//    把它拉起来（这正是 autorestart 的意义）。要真的停就用 pm2 stop。
//
// ⚠️ 不要同时用控制面板“直接启动”和 pm2：两个进程抢 2345 端口 + 抢 21.5 GiB
//    显存 → 必然 CUDA OOM。控制面板已改成检测到 pm2 托管时自动转发给 pm2。
// ===========================================================================

const path = require("path");

const APP_NAME = "qwen38-server";

module.exports = {
  apps: [
    {
      name: APP_NAME,

      // Windows 上不能直接用 .venv\Scripts\python.exe：
      // 那是个“转发器”，会再拉起 base 解释器，一次启动变成两个进程。
      // pm2 只认得转发器的 PID，杀掉它之后真正占显存的子进程会变成孤儿
      // （显存不释放，下次启动直接 OOM）。
      // 所以这里用 scripts\run_server.py 包一层，由它来做进程组管理。
      script: path.join(__dirname, "scripts", "run_server.py"),
      interpreter: path.join(__dirname, ".venv", "Scripts", "python.exe"),

      cwd: __dirname,
      args: [],

      // ---------------- 环境变量 ----------------
      // PYTHONIOENCODING / PYTHONUTF8 必须设：子进程重定向到文件时，
      // Windows 默认用系统 ANSI（GBK）写，日志会变成乱码。
      // PYTHONUNBUFFERED 让日志实时刷进文件，否则 pm2 logs 半天不出东西。
      env: {
        PYTHONIOENCODING: "utf-8:replace",
        PYTHONUTF8: "1",
        PYTHONUNBUFFERED: "1",
        // 让 run_server.py 直接把日志写进 logs/service.log（和控制面板同一份），
        // 而不是让 pm2 各自记一份，免得两处日志对不上。
        QWEN38_LOG_TO_FILE: "1",
      },

      // ---------------- 崩溃恢复策略 ----------------
      // 服务要吃掉 21.5 GiB 显存，重启代价很大（要重新加载权重 + 预热 CUDA 图）。
      // 所以这里刻意做得**保守**：宁可停下来让人看一眼，也不要疯狂重启把 GPU 打爆。
      //
      // ⚠️ 参数不能乱调。pm2 判定「重启超限」的源码逻辑是
      //   （lib/God.js:411-423）：
      //       if (now - created_at < min_uptime * max_restarts) {
      //           if (now - pm_uptime < min_uptime) unstable_restarts += 1;
      //       }
      //       if (unstable_restarts >= max_restarts) → 置为 ERRORED，不再重启
      //
      //   所以真正决定宽容度的是乘积 min_uptime * max_restarts —— 它是
      //   「从应用创建算起」的一个时间窗。窗内每次「没活够 min_uptime 的重启」
      //   都会消耗一个名额，用完就 ERRORED。
      //
      //   这里取 30s * 10 = 300s：窗口内允许 10 次失败尝试。
      //   为什么不是更大/更小：
      //     - min_uptime 必须大于加载耗时（实测 7-15s），否则「加载中就崩」
      //       不会被判为失败；30s 给了两倍余量。
      //     - 窗口内允许 10 次，是为了让**正常切换配置的连续重启**不至于
      //       把名额烧光（切一次配置就是一次重启，模型要 10s 左右加载完）。
      //       早先设成 90s * 5 = 450s 窗口 / 只有 5 个名额，密集切配置会误判。
      //
      //   万一真的进了 ERRORED：状态会显示 errored，`pm2 start qwen38-server`
      //   即可恢复（控制面板的「启动」走的就是这条路）。
      autorestart: true,
      // 退避重启：每次失败后等待时间翻倍，避免“秒崩秒起”的循环
      exp_backoff_restart_delay: 15000,   // 起点 15s，之后 15/30/60/120...
      // 进程活够 30s 才算“这次启动是成功的”
      min_uptime: 30000,
      // 连续失败这么多次就放弃（状态变 errored），不再尝试
      max_restarts: 10,
      // 加载权重要时间，给足停止/重启的宽限期
      kill_timeout: 30000,
      // 不要在文件变动时重启（这个服务不是热重载的）
      watch: false,

      // ---------------- 日志 ----------------
      // 注意：这里只是 pm2 自己的记录。真正的服务日志由 run_server.py 写进
      // logs/service.log（受 QWEN38_LOG_TO_FILE 控制），控制面板读的是那一份。
      output: path.join(__dirname, "logs", "pm2-out.log"),
      error: path.join(__dirname, "logs", "pm2-err.log"),
      merge_logs: true,
      time: true,          // 每行加时间戳

      // 不要设 max_memory_restart：它看的是进程 RSS，而模型的显存在 GPU 上，
      // 设了只会在权重加载时误杀（加载期间 CPU RSS 也会涨）。
    },
  ],
};
