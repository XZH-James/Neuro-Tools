import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sys
from scipy.signal import find_peaks

# 设置中文字体以避免警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

def load_first_column(path):
    """
    逐行读取文件，尝试将每行第一列转换为 float，跳过无法转换的行。
    """
    vals = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                vals.append(float(parts[0]))
            except ValueError:
                continue
    return np.array(vals)

class SignalComparer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("刺激 vs. 钙荧光 可视化")
        self.geometry("900x700")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 文件选择区
        frm = tk.Frame(self)
        frm.pack(pady=10)
        self.stim_path = tk.StringVar()
        self.fluo_path = tk.StringVar()

        tk.Button(frm, text="选择 刺激 信号", command=self.select_stim).grid(row=0, column=0)
        tk.Label(frm, textvariable=self.stim_path, width=60, anchor="w").grid(row=0, column=1)

        tk.Button(frm, text="选择 荧光 信号", command=self.select_fluo).grid(row=1, column=0, pady=5)
        tk.Label(frm, textvariable=self.fluo_path, width=60, anchor="w").grid(row=1, column=1)

        # 时间轴长度输入框
        tk.Label(self, text="时间轴长度 (ms)：").pack()
        self.time_entry = tk.Entry(self)
        self.time_entry.insert(0, "45000")  # 默认45000ms
        self.time_entry.pack(pady=5)

        # 绘制按钮
        tk.Button(self, text="绘制信号", command=self.plot).pack(pady=10)

        # 校准按钮
        self.calibrate_button = tk.Button(self, text="校准刺激信号", command=self.calibrate_stim, state=tk.DISABLED)
        self.calibrate_button.pack(pady=10)

        # 保存按钮
        tk.Button(self, text="保存平移后的刺激信号", command=self.save_translated_stim).pack(pady=10)

        # Matplotlib 图表区
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))  # 两个子图
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()

        # 交互工具栏（缩放、平移等）
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        widget = self.canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 设置变量存储黄点位置
        self.yellow_point_time = None
        self.selected_fluo_peak = None
        self.translated_stim = None
        self.fluo = None  # 存储荧光信号

    def select_stim(self):
        p = filedialog.askopenfilename(
            title="选择刺激信号 TXT 文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*")]
        )
        if p:
            self.stim_path.set(p)

    def select_fluo(self):
        p = filedialog.askopenfilename(
            title="选择荧光信号 TXT 文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*")]
        )
        if p:
            self.fluo_path.set(p)

    def plot(self):
        stim_file = self.stim_path.get()
        fluo_file = self.fluo_path.get()
        if not stim_file or not fluo_file:
            messagebox.showwarning("缺少文件", "请先选择两个信号文件！")
            return

        try:
            duration_ms = float(self.time_entry.get())  # 用户输入的时间轴长度
            if duration_ms <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("输入错误", "时间轴长度必须是正数！")
            return

        stim = load_first_column(stim_file)
        self.fluo = load_first_column(fluo_file)
        if stim.size == 0 or self.fluo.size == 0:
            messagebox.showerror("读取失败", "至少有一个文件未读到有效数值！")
            return

        # 生成统一的时间轴
        t_stim = np.linspace(0, duration_ms, len(stim))  # 刺激信号对应的时间轴
        t_fluo = np.linspace(0, duration_ms, len(self.fluo))  # 荧光信号对应的时间轴

        # 清空图表
        self.ax1.clear()
        self.ax2.clear()

        # **荧光信号峰值检测**：使用动态阈值
        mean_fluo = np.mean(self.fluo)
        std_fluo = np.std(self.fluo)
        dynamic_threshold_fluo = mean_fluo + 2 * std_fluo  # 动态阈值：均值 + 2 标准差
        fluo_peaks, _ = find_peaks(self.fluo, height=dynamic_threshold_fluo)  # 查找峰值，使用动态阈值

        # **刺激信号峰值检测**：设定阈值为1，检测刺激信号峰值
        stim_peaks, _ = find_peaks(stim, height=1)  # 查找刺激信号峰值，阈值为 1

        # **绘制原始刺激信号和荧光信号**
        self.ax1.plot(t_stim, stim, label='原始刺激信号', color='tab:blue')
        self.ax1.plot(t_fluo, self.fluo, label='荧光信号', color='tab:red', alpha=0.7)

        # 标记刺激信号的峰值（绿色）
        self.ax1.scatter(t_stim[stim_peaks], stim[stim_peaks], color='green', label='刺激信号峰值', zorder=5)

        # 设置点击事件来选定黄点位置
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # 设置标题和标签
        self.ax1.set_xlabel('时间 (ms)')
        self.ax1.set_ylabel('信号值')
        self.ax1.set_title(f"原始信号对比")
        self.ax1.legend()

        self.fig.tight_layout()
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax1:
            return  # 如果点击不是在第一个子图内，则忽略

        # 获取点击位置的时间和信号值
        if event.xdata is not None and event.ydata is not None:
            clicked_time = event.xdata
            clicked_value = event.ydata

            # 找到最近的荧光信号峰
            fluo_peak_idx = np.argmin(np.abs(np.linspace(0, len(self.ax1.get_lines()[1].get_data()[0]), len(self.ax1.get_lines()[1].get_data()[0])) - clicked_time))
            self.selected_fluo_peak = fluo_peak_idx  # 设置黄点对应的荧光信号峰索引
            self.yellow_point_time = clicked_time

            # 在图上标记黄点
            self.ax1.scatter(clicked_time, clicked_value, color='yellow', label="黄点", zorder=5)
            self.fig.canvas.draw()

            # 激活“校准刺激信号”按钮
            self.calibrate_button.config(state=tk.NORMAL)

    def calibrate_stim(self):
        if self.yellow_point_time is None:
            messagebox.showwarning("没有选择黄点", "请点击荧光信号设置黄点！")
            return

        stim_file = self.stim_path.get()
        stim = load_first_column(stim_file)
        stim_peaks, _ = find_peaks(stim, height=1)  # 查找刺激信号峰值，阈值为 1
        if len(stim_peaks) == 0:
            messagebox.showwarning("没有峰值", "刺激信号中未找到有效峰值！")
            return

        # 根据黄点时间平移刺激信号
        duration_ms = float(self.time_entry.get())
        t_stim = np.linspace(0, duration_ms, len(stim))

        stim_peak_time = t_stim[stim_peaks[0]]  # 取第一个峰值进行校准
        shift_time = self.yellow_point_time - stim_peak_time - 150  # 偏移150ms
        time_step_stim = t_stim[1] - t_stim[0]  # 刺激信号时间步长
        shift_samples = int(shift_time / time_step_stim)
        print(self.yellow_point_time)
        print(stim_peak_time)
        print(shift_time)
        print(shift_samples)

        # 平移刺激信号
        self.translated_stim = np.copy(stim)
        if shift_samples > 0:
            self.translated_stim[shift_samples:] = stim[:-shift_samples]
            self.translated_stim[:shift_samples] = 0
        elif shift_samples < 0:
            self.translated_stim[:shift_samples] = stim[-shift_samples:]
            self.translated_stim[shift_samples:] = 0

        # 在第二个子图上绘制平移后的刺激信号，并标记灰色区域
        t_stim = np.linspace(0, float(self.time_entry.get()), len(self.translated_stim))
        self.ax2.plot(t_stim, self.translated_stim, label='平移后的刺激信号', color='tab:orange')

        # 绘制荧光信号
        t_fluo = np.linspace(0, float(self.time_entry.get()), len(self.fluo))
        self.ax2.plot(t_fluo, self.fluo, label='荧光信号', color='tab:red', alpha=0.7)

        # 绘制灰色区域表示平移部分
        self.ax2.fill_between(t_stim, self.translated_stim, stim, color='gray', alpha=0.3, label='平移区域')

        # 设置图标
        self.ax2.set_xlabel('时间 (ms)')
        self.ax2.set_ylabel('信号值')
        self.ax2.set_title("平移后的刺激信号与荧光信号")
        self.ax2.legend()

        self.fig.tight_layout()
        self.canvas.draw()

    def save_translated_stim(self):
        if self.translated_stim is None:
            messagebox.showwarning("没有平移信号", "请先校准刺激信号！")
            return

        save_path = filedialog.asksaveasfilename(
            title="保存平移后的刺激信号",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*")],
            initialfile="data_user_input_new.txt"
        )
        if save_path:
            np.savetxt(save_path, self.translated_stim, fmt="%.6f")
            messagebox.showinfo("保存成功", f"平移后的刺激信号已保存至 {save_path}")

    def on_closing(self):
        self.destroy()
        sys.exit()

if __name__ == '__main__':
    app = SignalComparer()
    app.mainloop()
