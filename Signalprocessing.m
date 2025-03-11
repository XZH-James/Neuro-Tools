%% 主脚本部分
% for Fluorescence signal processing

% --- 数据加载 --- %load the Fluorescence signal file .txt
dataPath = '\.txt'; 
CalciumData = readmatrix(dataPath);

% 检查数据尺寸并转置（确保数据为 1×T）
disp('数据尺寸:');
disp(size(CalciumData));
if size(CalciumData, 1) > size(CalciumData, 2)
    CalciumData = CalciumData';
end

% --- 参数设置 ---
Parameters.SamplingRate = 20;         % 采样率（Hz）
Parameters.HighThreshold = 1;         % 高阈值系数（根据需要调整）
Parameters.LowThreshold = 0;          % 低阈值系数
Parameters.PredictionLength = 0.5;      % 预测窗口长度（秒）
Parameters.BaselineLength = 1;          % 用于基线估计的窗口长度（秒）
Parameters.LPfactor = 0.1;              % 平滑因子
Parameters.UseDetection = true;       % 启用事件检测

% --- 局部基线校正 ---
% 假设信号大部分时间处于基线状态，使用中值滤波估计局部基线
cellIndex = 1;  % 处理单个神经元信号
window_size = floor(Parameters.BaselineLength * Parameters.SamplingRate);
baseline_estimate = medfilt1(CalciumData(cellIndex, :), window_size);
% 对信号进行校正，使得大部分时间的基线接近于0
CalciumData_corrected = CalciumData(cellIndex, :) - baseline_estimate;

% --- 基于校正后信号重新进行事件检测 ---
% 这里我们将校正后的信号作为输入进行事件检测，得到的检测索引即对应校正后信号
DetectedEvents_corrected = TransientDetection(CalciumData_corrected, Parameters);

% --- 绘图显示 ---
figure;

% 绘制原始信号及其局部基线（仅供参考）
subplot(3,1,1);
plot(CalciumData(cellIndex, :), 'b-'); hold on;
plot(baseline_estimate, 'k--', 'LineWidth', 2);
legend('原始信号','局部基线');
title(['原始信号及局部基线 (Cell ' num2str(cellIndex) ')']);

% 绘制校正后的信号
subplot(3,1,2);
plot(CalciumData_corrected, 'b-'); hold on;
plot(zeros(size(CalciumData_corrected)), 'k--', 'LineWidth', 2);
legend('校正后信号','水平基线 (0)');
title('局部基线校正后的信号');

% 绘制校正后信号及事件检测结果
subplot(3,1,3);
plot(CalciumData_corrected, 'b-'); hold on;
if ~isempty(DetectedEvents_corrected.OnsetDetectionResult{1})
    plot(DetectedEvents_corrected.OnsetDetectionResult{1}, ...
         CalciumData_corrected(DetectedEvents_corrected.OnsetDetectionResult{1}), 'go', 'MarkerSize', 10);
end
if ~isempty(DetectedEvents_corrected.PeakDetectionResult{1})
    plot(DetectedEvents_corrected.PeakDetectionResult{1}, ...
         CalciumData_corrected(DetectedEvents_corrected.PeakDetectionResult{1}), 'ro', 'MarkerSize', 10);
end
if ~isempty(DetectedEvents_corrected.EndDetectionResult{1})
    plot(DetectedEvents_corrected.EndDetectionResult{1}, ...
         CalciumData_corrected(DetectedEvents_corrected.EndDetectionResult{1}), 'ko', 'MarkerSize', 10);
end
legend('校正后信号','起始点','峰值','结束点');
title('校正后信号及事件检测结果');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TransientDetection 函数定义
function DetectedEvents = TransientDetection(CalciumData, Parameters)
% TransientDetection - 对神经元信号进行瞬态事件检测
%
% 输入：
%   CalciumData - 数值矩阵，每一行为一个神经元信号，列为时间点（这里传入的是校正后的信号）
%   Parameters  - 参数结构体，包含：
%                   SamplingRate: 采样率（Hz）
%                   HighThreshold: 高阈值系数
%                   LowThreshold: 低阈值系数
%                   PredictionLength: 预测窗口长度（秒）
%                   BaselineLength: 基线窗口长度（秒）
%                   LPfactor: 平滑因子
%                   UseDetection: 是否启用事件检测（true/false）
%
% 输出：
%   DetectedEvents - 结构体，包含各个检测结果字段

    % 从 Parameters 中获取参数
    SamplingRate = Parameters.SamplingRate;
    HighThresholdFactor = Parameters.HighThreshold;
    RateThreshold = Parameters.LowThreshold;
    PredictWindow = floor(Parameters.PredictionLength * SamplingRate);
    BaselineWindow = floor(Parameters.BaselineLength * SamplingRate);
    LPfactor = Parameters.LPfactor;
    % 设置峰值窗口参数（单位：采样点）
    PeakAfterPoint = floor(0.1 * SamplingRate);
    PeakPrePoint = floor(0.1 * SamplingRate);
    EndWindow = floor(0.1 * SamplingRate);
    PeakWindow = floor(0.1 * SamplingRate);
    PeakPoint = floor(0.1 * SamplingRate);
    DecayPoint = floor(0.5 * SamplingRate);
    LPFvalue = floor(LPfactor * SamplingRate);

    if Parameters.UseDetection
        for cell_index = 1:size(CalciumData, 1)
            DataOfSingleCell = CalciumData(cell_index, :);
            DataOfSingleCellSmoothed = movmean(DataOfSingleCell, LPFvalue);
            Residual = DataOfSingleCell - DataOfSingleCellSmoothed;
            
            OnsetDetectionResult{cell_index} = [];
            PeakDetectionResult{cell_index} = [];
            EndDetectionResult{cell_index} = [];
            ThresholdValue{cell_index} = [];
            
            sample_index = 1;
            while sample_index <= (length(DataOfSingleCell) - (BaselineWindow + PredictWindow) + 1)
                % 取数据片段
                TestData = DataOfSingleCell(sample_index+BaselineWindow : sample_index+BaselineWindow+PredictWindow-1);
                DataOfBaseline = DataOfSingleCell(sample_index : sample_index+BaselineWindow-1);
                TestDataSmoothed = DataOfSingleCellSmoothed(sample_index+BaselineWindow : sample_index+BaselineWindow+PredictWindow-1);
                DataOfBaselineSmoothed = DataOfSingleCellSmoothed(sample_index : sample_index+BaselineWindow-1);
                % 基线校正（一次多项式拟合）
                cofactor = polyfit(1:BaselineWindow, DataOfBaseline, 1);
                BaselineCorrection = (1:BaselineWindow)*cofactor(1) + cofactor(2);
                DataOfBaseline_corrected = DataOfBaseline - BaselineCorrection + mean(BaselineCorrection);
                % 计算基线校正后的测试数据
                TestDataCorrected = TestData - mean(DataOfBaseline);
                TestDataCorrectedSmoothed = TestDataSmoothed - mean(DataOfBaselineSmoothed);
                [~, MaxTestPoint] = max(TestDataCorrected);
                if MaxTestPoint > PeakPrePoint && MaxTestPoint <= (length(TestDataCorrected) - PeakAfterPoint)
                    TestValue = mean(TestDataCorrected(MaxTestPoint-PeakPrePoint : MaxTestPoint+PeakAfterPoint));
                elseif MaxTestPoint > PeakPrePoint && MaxTestPoint > (length(TestDataCorrected) - PeakAfterPoint)
                    TestValue = mean(TestDataCorrected(MaxTestPoint-PeakPrePoint : end));
                elseif MaxTestPoint <= PeakPrePoint && MaxTestPoint <= (length(TestDataCorrected) - PeakAfterPoint)
                    TestValue = mean(TestDataCorrected(1 : MaxTestPoint+PeakAfterPoint));
                elseif MaxTestPoint <= PeakPrePoint && MaxTestPoint > (length(TestDataCorrected) - PeakAfterPoint)
                    TestValue = mean(TestDataCorrected(1 : end));
                end
                % 判断标准：峰值和上升速率阈值
                BaselineSTD = std(DataOfBaseline_corrected);
                Criterion1 = TestValue > (HighThresholdFactor * BaselineSTD);
                TestDataCorrectedDifference = diff(TestDataCorrected);
                MaxDifference = max(TestDataCorrectedDifference);
                Criterion2 = MaxDifference > (RateThreshold * BaselineSTD);
                
                [~, MaxSmoothedPoint] = max(TestDataCorrectedSmoothed);
                [~, MaxPoint] = max(TestDataCorrected);
                
                if Criterion1 && Criterion2
                    ValueTorecord = [TestValue; MaxDifference; BaselineSTD];
                    ThresholdValue{cell_index} = [ThresholdValue{cell_index}, ValueTorecord];
                    DataCombined = TestDataCorrected;
                    % 定位起始点
                    if length(DataCombined) > 1
                        DataCombined_diff = diff(DataCombined);
                        if length(find(DataCombined_diff > (RateThreshold * BaselineSTD))) > 1
                            fast_point_first = find(DataCombined_diff > (RateThreshold * BaselineSTD));
                            fast_point = fast_point_first(1);
                            for fp = 1:length(fast_point_first)-1
                                if (DataCombined(fast_point_first(fp)+1) < DataCombined(fast_point_first(fp+1)+1)) && ...
                                   (fast_point_first(fp+1)-fast_point_first(fp) > 1) && ...
                                   (DataCombined(fast_point_first(fp)+2) < DataCombined(fast_point_first(fp)+1))
                                    fast_point = fast_point_first(fp+1);
                                    break;
                                end
                            end
                        else
                            fast_point = find(DataCombined_diff > (RateThreshold * BaselineSTD));
                        end
                        OnsetDetected = sample_index - 1 + BaselineWindow + fast_point;
                    else
                        OnsetDetected = sample_index - 1 + BaselineWindow;
                    end;
                    
                    n = 1;
                    if (BaselineWindow + sample_index - 1 + MaxPoint + n + PeakPoint <= length(DataOfSingleCell))
                        while (mean(DataOfSingleCell(BaselineWindow + sample_index - 1 + MaxPoint + n : BaselineWindow + sample_index - 1 + MaxPoint + n + PeakPoint - 1)) - ...
                               DataOfSingleCell(BaselineWindow + sample_index - 1 + MaxPoint + n - 1)) > 0
                            n = n + 1;
                            if (BaselineWindow + sample_index - 1 + MaxPoint + n + PeakPoint > length(DataOfSingleCell))
                                break;
                            end
                        end
                    end
                    PeakDetected = BaselineWindow + sample_index - 1 + MaxPoint + n - 1;
                    
                    j = 1;
                    if (BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j + PeakWindow <= length(DataOfSingleCellSmoothed))
                        while (DataOfSingleCellSmoothed(BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j) - ...
                               DataOfSingleCellSmoothed(BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j - 1)) > 0 || ...
                              (mean(DataOfSingleCellSmoothed(BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j : BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j + PeakWindow)) - ...
                               DataOfSingleCellSmoothed(BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j)) > 0
                            j = j + 1;
                            if (BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j + PeakWindow > length(DataOfSingleCellSmoothed))
                                break;
                            end
                        end
                    end
                    
                    if (BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j - 1 + PeakWindow) <= length(DataOfSingleCell)
                        [~, peak_max_ind] = max(DataOfSingleCell(BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j - 1 - PeakWindow + 1 : ...
                                                                   BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j - 1 + PeakWindow));
                    else
                        [~, peak_max_ind] = max(DataOfSingleCell(BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j - 1 - PeakWindow + 1 : end));
                    end
                    
                    Peak_smoothed = BaselineWindow + sample_index - 1 + MaxSmoothedPoint + j - 1;
                    
                    p = 1;
                    if (Peak_smoothed + p + EndWindow <= length(DataOfSingleCellSmoothed))
                        while (max(DataOfSingleCellSmoothed(Peak_smoothed + p + 1 : Peak_smoothed + EndWindow + p)) > HighThresholdFactor * BaselineSTD) && ...
                              mean(diff(DataOfSingleCellSmoothed(Peak_smoothed + p + 1 : Peak_smoothed + EndWindow + p))) < 0
                            p = p + 1;
                            if (Peak_smoothed + EndWindow + p > length(DataOfSingleCellSmoothed))
                                break;
                            end
                        end
                    end
                    EndDetected_smoothed = Peak_smoothed + p + 1;
                    if EndDetected_smoothed + EndWindow <= length(DataOfSingleCellSmoothed)
                        [~, ind_diff_min] = min(DataOfSingleCell(EndDetected_smoothed + 1 : EndDetected_smoothed + EndWindow));
                    else
                        [~, ind_diff_min] = min(DataOfSingleCell(EndDetected_smoothed + 1 : end));
                    end
                    EndDetected = EndDetected_smoothed + ind_diff_min;
                    
                    if PeakDetected + DecayPoint > length(DataOfSingleCell)
                        DecayValue = DataOfSingleCell(PeakDetected : end);
                    else
                        DecayValue = DataOfSingleCell(PeakDetected : PeakDetected + DecayPoint);
                    end
                    if mean(DecayValue) < RateThreshold * BaselineSTD
                        sample_index = EndDetected - BaselineWindow + 1;
                        continue;
                    end
                    
                    if ~isempty(EndDetectionResult) && ~isempty(OnsetDetectionResult)
                        ReplaceWindow = BaselineWindow + 1 : BaselineWindow + (EndDetected - OnsetDetected) + 1;
                        if length(ReplaceWindow) >= BaselineWindow
                            ReplaceData = Residual(OnsetDetected : EndDetected) + DataOfSingleCell(EndDetected);
                            DataOfSingleCell(OnsetDetected : EndDetected) = ReplaceData;
                        else
                            ReplaceData = Residual(EndDetected - BaselineWindow + 1 : EndDetected) + DataOfSingleCell(EndDetected);
                            DataOfSingleCell(EndDetected - BaselineWindow + 1 : EndDetected) = ReplaceData;
                        end
                    end
                    OnsetDetectionResult{cell_index} = [OnsetDetectionResult{cell_index}, OnsetDetected];
                    PeakDetectionResult{cell_index} = [PeakDetectionResult{cell_index}, PeakDetected];
                    EndDetectionResult{cell_index} = [EndDetectionResult{cell_index}, EndDetected];
                    
                    sample_index = EndDetected - BaselineWindow + 1;
                else
                    sample_index = sample_index + 1;
                end
            end
        end
        DetectionNeeded = 0;
    else
        for cell_index = 1:size(CalciumData, 1)
            OnsetDetectionResult{cell_index} = [];
            PeakDetectionResult{cell_index} = [];
            EndDetectionResult{cell_index} = [];
        end
        DetectionNeeded = 1;
    end

    DetectedEvents.OnsetDetectionResult = OnsetDetectionResult;
    DetectedEvents.PeakDetectionResult  = PeakDetectionResult;
    DetectedEvents.EndDetectionResult   = EndDetectionResult;
    if Parameters.UseDetection
        DetectedEvents.ThresholdValue = ThresholdValue;
    end
    DetectedEvents.DetectionNeeded = DetectionNeeded;
end
