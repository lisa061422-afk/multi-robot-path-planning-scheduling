function fig = plot_coarse_schedule_from_mat(matFile, varargin)
%PLOT_COARSE_SCHEDULE_FROM_MAT Plot coarse local schedules from Python .mat data.
%
% Data expected in matFile:
%   segments columns:
%     [vehicle_id, task_index, resource, requested_time, start_time,
%      end_time, delay, turn_code]
%   coords columns:
%     [resource, x, y]
%
% turn_code: 1=left, 2=straight, 3=right.
%
% Example:
%   fig = plot_coarse_schedule_from_mat('output/coarse_schedule_data.mat');
%   fig = plot_coarse_schedule_from_mat('output/coarse_schedule_data.mat', ...
%       'outputFile', 'output/coarse_schedule_panel.pdf');

if nargin < 1 || isempty(matFile)
    matFile = fullfile('output', 'coarse_schedule_data.mat');
end

p = inputParser;
addParameter(p, 'outputFile', '', @(s) ischar(s) || isstring(s));
addParameter(p, 'figColor', [1 1 1], @(x) isnumeric(x) && numel(x)==3);
addParameter(p, 'panelColor', [1 1 1], @(x) isnumeric(x) && numel(x)==3);
addParameter(p, 'axColor', [1 1 1], @(x) isnumeric(x) && numel(x)==3);
addParameter(p, 'lineColor', [0.35 0.75 0.95], @(x) isnumeric(x) && numel(x)==3);
addParameter(p, 'delayColor', [0.75 0.75 0.75], @(x) isnumeric(x) && numel(x)==3);
addParameter(p, 'hatchSpacing', 0.18, @(x) isscalar(x) && x > 0);
addParameter(p, 'spaceColors', [ ...
    0.16 0.95 0.16; ...
    0.98 0.84 0.00; ...
    1.00 0.55 0.00; ...
    0.12 0.66 0.84; ...
    0.95 0.10 0.10; ...
    0.55 0.45 0.95], @(x) isnumeric(x) && size(x,2)==3);
parse(p, varargin{:});
opt = p.Results;

D = load(matFile);
segments = D.segments;
coords = D.coords;

if isempty(segments)
    error('segments is empty; no schedule to plot.');
end

resources = coords(:,1)';
xs = coords(:,2)';
ys = coords(:,3)';
minX = min(xs); maxX = max(xs);
minY = min(ys); maxY = max(ys);
nCol = maxX - minX + 1;
nRow = maxY - minY + 1;

fig = figure('Color', opt.figColor, 'Units', 'normalized', ...
    'Position', [0.12 0.14 0.62 0.66]);

outerGap = 0.025;
margX = 0.035;
margY = 0.050;
cellW = (1 - 2*margX - (nCol-1)*outerGap) / nCol;
cellH = (1 - 2*margY - (nRow-1)*outerGap) / nRow;

for rr = resources
    row = maxY - coords(coords(:,1)==rr,3) + 1;
    col = coords(coords(:,1)==rr,2) - minX + 1;
    left = margX + (col-1)*(cellW + outerGap);
    bottom = 1 - margY - row*cellH - (row-1)*outerGap;

    panel = uipanel('Parent', fig, 'Units', 'normalized', ...
        'Position', [left bottom cellW cellH], ...
        'BackgroundColor', opt.panelColor, ...
        'BorderType', 'line', 'Title', sprintf('M%d', rr), ...
        'FontWeight', 'bold', 'FontSize', 12);

    localSeg = segments(segments(:,3)==rr, :);
    draw_one_resource_panel(panel, localSeg, rr, opt);
end

sgtitle(fig, 'Coarse Local Schedule Panel', 'FontSize', 16, 'FontWeight', 'bold');

if strlength(string(opt.outputFile)) > 0
    exportgraphics(fig, opt.outputFile, 'ContentType', 'vector');
end
end

function draw_one_resource_panel(panel, segs, resource, opt)
if isempty(segs)
    ax = axes('Parent', panel, 'Units', 'normalized', 'Position', [0.12 0.20 0.82 0.62]);
    axis(ax, 'off');
    text(ax, 0.5, 0.5, 'No scheduled local task', ...
        'HorizontalAlignment', 'center', 'Color', [0.4 0.45 0.5]);
    return;
end

veh = unique(segs(:,1)', 'stable');
Nv = numel(veh);
Nax = Nv + 1;
gap = 0.025;
margH = 0.12;
margW = 0.22;
titlePad = 0.04;
axH = (1 - 2*margH - titlePad - (Nax-1)*gap) / Nax;
axW = 1 - 2*margW;

xmin = min([segs(:,4); segs(:,5)]) - 0.4;
xmax = max(segs(:,6)) + 0.4;
if xmax <= xmin
    xmax = xmin + 1;
end

for ii = 1:Nv
    bottom = 1 - margH - titlePad - ii*axH - (ii-1)*gap;
    ax = axes('Parent', panel, 'Units', 'normalized', ...
        'Position', [margW bottom axW axH]);
    setup_axis(ax, opt, xmin, xmax);

    v = veh(ii);
    vSegs = segs(segs(:,1)==v, :);

    draw_stairs_from_segments(ax, vSegs, xmin, xmax, opt.lineColor, true);

    for k = 1:size(vSegs,1)
        s = vSegs(k,:);
        req = s(4); st = s(5); en = s(6); task = s(2); turnCode = s(8);
        if st > req
            draw_delay_hatch(ax, req, st, 0, 1, opt.delayColor, opt.hatchSpacing);
        end
        color = color_for_resource(resource, opt.spaceColors);
        rectangle(ax, 'Position', [st 0 en-st 1], ...
            'FaceColor', color, 'EdgeColor', 'none');
        text(ax, st + (en-st)/2, 0.5, sprintf('C%d %s, K%d', task, turn_name(turnCode), task), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 10, 'FontWeight', 'bold', 'Clipping', 'on');
    end

    text(ax, xmin - 0.16*(xmax-xmin), 0.5, sprintf('N%d', v), ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Clipping', 'off');
    if ii < Nax
        set(ax, 'XTickLabel', []);
    end
end

bottom = 1 - margH - titlePad - Nax*axH - (Nax-1)*gap;
ax = axes('Parent', panel, 'Units', 'normalized', ...
    'Position', [margW bottom axW axH]);
setup_axis(ax, opt, xmin, xmax);
draw_stairs_from_segments(ax, segs, xmin, xmax, opt.lineColor, false);

color = color_for_resource(resource, opt.spaceColors);
for k = 1:size(segs,1)
    s = segs(k,:);
    v = s(1); st = s(5); en = s(6);
    rectangle(ax, 'Position', [st 0 en-st 1], ...
        'FaceColor', 'none', 'EdgeColor', color, 'LineWidth', 2.0);
    text(ax, st + (en-st)/2, 0.5, sprintf('N%d', v), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Clipping', 'on');
end
text(ax, xmin - 0.16*(xmax-xmin), 0.5, sprintf('M%d', resource), ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Clipping', 'off');
xlabel(ax, 'Time (seconds)', 'FontSize', 10);
end

function setup_axis(ax, opt, xmin, xmax)
hold(ax, 'on');
set(ax, 'Color', opt.axColor, 'FontSize', 10, 'Box', 'on', ...
    'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.16, 'Layer', 'bottom');
xlim(ax, [xmin xmax]);
ylim(ax, [0 1.2]);
yticks(ax, [0 0.5 1]);
end

function draw_stairs_from_segments(ax, segs, xmin, xmax, lineColor, includeDelay)
times = unique([xmin; xmax; segs(:,4); segs(:,5); segs(:,6)]);
times = sort(times(:));
y = zeros(size(times));
for i = 1:numel(times)
    t = times(i);
    active = any(t >= segs(:,5) & t < segs(:,6));
    waiting = includeDelay && any(t >= segs(:,4) & t < segs(:,5));
    if active
        y(i) = 1;
    elseif waiting
        y(i) = 0.5;
    else
        y(i) = 0;
    end
end
stairs(ax, times, y, 'Color', lineColor, 'LineWidth', 1.8);
end

function draw_delay_hatch(ax, x1, x2, y1, y2, lineColor, spacing)
if x2 <= x1 || y2 <= y1
    return;
end
hold(ax, 'on');
H = y2 - y1;
cVals = (x1 - H):spacing:x2;
for c = cVals
    xa = max(x1, c);
    xb = min(x2, c + H);
    if xb > xa
        ya = y1 + (xa - c);
        yb = y1 + (xb - c);
        plot(ax, [xa xb], [ya yb], '-', ...
            'Color', lineColor, 'LineWidth', 0.8, 'Clipping', 'on');
    end
end
plot(ax, [x1 x2 x2 x1 x1], [y1 y1 y2 y2 y1], '-', ...
    'Color', [0.82 0.82 0.82], 'LineWidth', 0.5);
end

function color = color_for_resource(resource, colors)
idx = mod(resource - 1, size(colors,1)) + 1;
color = colors(idx,:);
end

function name = turn_name(code)
names = {'left', 'straight', 'right'};
idx = max(1, min(3, round(code)));
name = names{idx};
end
