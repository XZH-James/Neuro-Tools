import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("数据可视化程序")
        self.geometry("1000x800")
        self.pressure_file = None
        self.fluorescence_file = None
        self.time_windows = []  # 保存所有生成的时间窗口，格式为 (start, end)
        self.t_pressure = None  # 压力曲线时间轴
        self.t_fluo = None  # 荧光曲线时间轴
        self.fluorescence_data = None  # 各神经元荧光数据，列表，每个元素为一维数组
        self.pressure_ax = None  # 压力曲线所在的 Axes（用于响应点击）
        self.axes = None  # 所有子图 Axes 列表
        self.canvas = None  # FigureCanvasTkAgg 对象
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 控制区域
        control_frame = tk.Frame(self)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        # 压力曲线文件选择
        btn_pressure = tk.Button(control_frame, text="选择压力曲线文件 (txt, xlsx)", command=self.load_pressure_file)
        btn_pressure.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lbl_pressure = tk.Label(control_frame, text="未选择文件")
        self.lbl_pressure.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # 荧光曲线文件选择
        btn_fluorescence = tk.Button(control_frame, text="选择荧光曲线文件 (txt, xlsx)", command=self.load_fluorescence_file)
        btn_fluorescence.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.lbl_fluorescence = tk.Label(control_frame, text="未选择文件")
        self.lbl_fluorescence.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # 压力采样频率设置
        lbl_pressure_freq = tk.Label(control_frame, text="压力采样频率(Hz):")
        lbl_pressure_freq.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_pressure_freq = tk.Entry(control_frame, width=10)
        self.entry_pressure_freq.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.entry_pressure_freq.insert(0, "800")

        # 荧光采样频率设置
        lbl_fluo_freq = tk.Label(control_frame, text="荧光采样频率(Hz):")
        lbl_fluo_freq.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_fluo_freq = tk.Entry(control_frame, width=10)
        self.entry_fluo_freq.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.entry_fluo_freq.insert(0, "40")

        # 时间窗口参数设置
        lbl_time_window = tk.Label(control_frame, text="时间窗口(秒):")
        lbl_time_window.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.entry_time_window = tk.Entry(control_frame, width=10)
        self.entry_time_window.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.entry_time_window.insert(0, "5")

        # 绘图按钮
        btn_plot = tk.Button(control_frame, text="绘制图形", command=self.plot_data)
        btn_plot.grid(row=5, column=0, columnspan=2, padx=5, pady=10)

        # 保存数据按钮
        btn_save = tk.Button(control_frame, text="保存数据", command=self.save_data)
        btn_save.grid(row=6, column=0, padx=5, pady=5, sticky="w")

        # 创建滚动显示区域（包含横向和纵向滚动条）
        self.plot_container = tk.Frame(self)
        self.plot_container.pack(side="top", fill="both", expand=True)

        self.plot_canvas = tk.Canvas(self.plot_container)
        self.plot_scrollbar = tk.Scrollbar(self.plot_container, orient="vertical", command=self.plot_canvas.yview)
        self.plot_canvas.configure(yscrollcommand=self.plot_scrollbar.set)
        self.plot_scrollbar.pack(side="right", fill="y")

        self.plot_h_scrollbar = tk.Scrollbar(self.plot_container, orient="horizontal", command=self.plot_canvas.xview)
        self.plot_canvas.configure(xscrollcommand=self.plot_h_scrollbar.set)
        self.plot_h_scrollbar.pack(side="bottom", fill="x")

        self.plot_canvas.pack(side="left", fill="both", expand=True)

        # 在 Canvas 中创建一个 Frame 用于放置图形
        self.plot_frame = tk.Frame(self.plot_canvas)
        self.plot_canvas.create_window((0, 0), window=self.plot_frame, anchor="nw")
        self.plot_frame.bind("<Configure>",
                             lambda event: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all")))

    def load_pressure_file(self):
        file_path = filedialog.askopenfilename(
            title="选择压力曲线文件",
            filetypes=[("Text files", "*.txt"), ("Excel files", "*.xlsx")])
        if file_path:
            self.pressure_file = file_path
            self.lbl_pressure.config(text=file_path)

    def load_fluorescence_file(self):
        file_path = filedialog.askopenfilename(
            title="选择荧光曲线文件",
            filetypes=[("Text files", "*.txt"), ("Excel files", "*.xlsx")])
        if file_path:
            self.fluorescence_file = file_path
            self.lbl_fluorescence.config(text=file_path)

    def read_file(self, file_path):
        if file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            try:
                return pd.read_csv(file_path, delim_whitespace=True, header=None, low_memory=False)
            except Exception:
                return pd.read_csv(file_path, sep=',', header=None, low_memory=False)

    def plot_data(self):
        # 重置时间窗口列表
        self.time_windows = []
        if not self.pressure_file or not self.fluorescence_file:
            messagebox.showwarning("文件未选择", "请先选择两个目标文件")
            return
        try:
            pressure_freq = float(self.entry_pressure_freq.get())
        except ValueError:
            pressure_freq = 800.0
            messagebox.showwarning("频率错误", "压力采样频率输入无效，使用默认800Hz")
        try:
            fluo_freq = float(self.entry_fluo_freq.get())
        except ValueError:
            fluo_freq = 40.0
            messagebox.showwarning("频率错误", "荧光采样频率输入无效，使用默认40Hz")

        try:
            df_pressure = self.read_file(self.pressure_file)
            if df_pressure.shape[1] < 3:
                messagebox.showerror("数据错误", "压力文件数据列数不足3列")
                return
            electrical_stim = pd.to_numeric(df_pressure.iloc[:, 1], errors='coerce').to_numpy()
            pressure_coef = pd.to_numeric(df_pressure.iloc[:, 2], errors='coerce').to_numpy()
            self.t_pressure = np.arange(len(electrical_stim)) / pressure_freq
        except Exception as e:
            messagebox.showerror("读取错误", f"读取压力文件失败:\n{e}")
            return

        try:
            df_fluorescence = self.read_file(self.fluorescence_file)
            # 每一列代表一个神经元的荧光曲线
            self.fluorescence_data = [pd.to_numeric(df_fluorescence.iloc[:, i], errors='coerce').to_numpy()
                                      for i in range(df_fluorescence.shape[1])]
            self.t_fluo = np.arange(df_fluorescence.shape[0]) / fluo_freq
        except Exception as e:
            messagebox.showerror("读取错误", f"读取荧光文件失败:\n{e}")
            return

        # 清除之前的图像，重新创建内部 frame
        self.plot_frame.destroy()
        self.plot_frame = tk.Frame(self.plot_canvas)
        self.plot_canvas.create_window((0, 0), window=self.plot_frame, anchor="nw")
        self.plot_frame.bind("<Configure>",
                             lambda event: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all")))

        # 子图数量：电刺激、压力曲线各1个 + 神经元荧光曲线（每列一个）
        num_subplots = 2 + len(self.fluorescence_data)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(80, num_subplots * 2), sharex=True)
        if num_subplots == 1:
            axes = [axes]
        self.axes = axes

        # 绘制第一幅子图：电刺激
        axes[0].plot(self.t_pressure, electrical_stim)
        axes[0].set_title("电刺激")
        axes[0].set_ylabel("幅值")
        axes[0].set_xlabel("时间 (s)")

        # 绘制第二幅子图：压力曲线（设置为响应点击的区域）
        axes[1].plot(self.t_pressure, pressure_coef)
        axes[1].set_title("压力系数")
        axes[1].set_ylabel("幅值")
        axes[1].set_xlabel("时间 (s)")
        self.pressure_ax = axes[1]

        # 绘制各个神经元的荧光曲线
        for idx, curve in enumerate(self.fluorescence_data):
            ax = axes[2 + idx]
            ax.plot(self.t_fluo, curve, linewidth=0.8)
            ax.set_title(f"神经元 {idx + 1}")
            ax.set_ylabel("荧光强度")
            ax.set_xlabel("时间 (s)")

        # 设置每个子图 x 轴主刻度（每 50 秒一个刻度）
        for ax in axes:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # 绑定点击事件：只在压力曲线所在 Axes 内响应点击
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        # 仅在压力曲线所在 Axes 内点击时响应
        if event.inaxes == self.pressure_ax and event.xdata is not None:
            clicked_time = event.xdata
            try:
                window_len = float(self.entry_time_window.get())
            except ValueError:
                window_len = 5.0
                messagebox.showwarning("参数错误", "时间窗口输入无效，使用默认5秒")
            window = (clicked_time, clicked_time + window_len)
            self.time_windows.append(window)
            # 在所有子图中添加灰色背景区域显示该时间窗口
            for ax in self.axes:
                ax.axvspan(window[0], window[1], facecolor='gray', alpha=0.3)
            self.canvas.draw()
            messagebox.showinfo("时间窗记录", f"记录时间窗: {window[0]:.2f} 到 {window[1]:.2f}秒")

    def save_data(self):
        # 提取各神经元在每个时间窗口下的荧光信号，不做平均处理
        if self.fluorescence_data is None or len(self.time_windows) == 0:
            messagebox.showwarning("数据不足", "请先生成时间窗后再保存数据")
            return
        segments = {}  # {neuron_index: [segment1, segment2, ...]}
        for i, signal in enumerate(self.fluorescence_data):
            segments[i] = []
            for window in self.time_windows:
                start, end = window
                indices = np.where((self.t_fluo >= start) & (self.t_fluo < end))[0]
                if len(indices) > 0:
                    seg = signal[indices]
                    segments[i].append(seg)
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx"), ("Text files", "*.txt")])
        if not file_path:
            return
        if file_path.endswith('.xlsx'):
            with pd.ExcelWriter(file_path) as writer:
                for i, seg_list in segments.items():
                    if len(seg_list) == 0:
                        continue
                    # 对于每个神经元，将每个时间窗口下的信号分别保存到一列中
                    min_len = min(len(seg) for seg in seg_list)
                    data = {}
                    for j, seg in enumerate(seg_list):
                        data[f'window_{j + 1}'] = seg[:min_len]
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name=f'Neuron_{i + 1}', index=False)
            messagebox.showinfo("保存成功", f"数据已保存到 {file_path}")
        else:
            with open(file_path, 'w') as f:
                for i, seg_list in segments.items():
                    if len(seg_list) == 0:
                        continue
                    min_len = min(len(seg) for seg in seg_list)
                    f.write(f'Neuron {i + 1}\n')
                    for j, seg in enumerate(seg_list):
                        f.write(f'Window {j + 1}:\n')
                        np.savetxt(f, seg[:min_len], fmt='%f')
                    f.write('\n')
            messagebox.showinfo("保存成功", f"数据已保存到 {file_path}")

    def on_closing(self):
        plt.close('all')
        self.destroy()


if __name__ == "__main__":
    app = DataVisualizer()
    app.mainloop()
