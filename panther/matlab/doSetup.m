function doSetup()
set(0,'DefaultFigureWindowStyle','docked') %'normal' 'docked'
set(0,'defaulttextInterpreter','latex'); set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultfigurecolor',[1 1 1])

set(0, 'DefaultFigureRenderer', 'painters'); %or opengl

addpath(genpath('./../../submodules/minvo/src/utils'));
addpath(genpath('./../../submodules/minvo/src/solutions'));
addpath(genpath('./more_utils'));

end