%%
load('test_for_filter.mat');
x = data(:, 2);
t = data(:, 1);

%%
l = length(x);
f = Fs * ((-l/2 + 1):(l/2))/l;
subplot(2, 1, 1)
plot(t, x, 'b')
xlabel('t (s)')
ylabel('x(t)')
xlim([0, t(end)])
subplot(2, 1, 2)
plot(f, fftshift(abs(fft(x))), 'r')
ylim([0, 3])
xlabel('f (Hz)')
ylabel('X(f)')

%%
plot(diff(data(:, 1)), 'b')
hold on
plot([0, length(data(:, 1))], [1, 1] * mean(diff(data(:, 1))), 'r')
xlim([0, length(data(:, 1))])
xlabel('sample')
legend('time between samples', 'mean')
Fs = 1/mean(diff(t));

%%
FIR = [0.0011422, 0.0051294, 0.008849, 0.011549, 0.012542, 0.011337, 0.0077619, 0.0020454, -0.0051582, -0.012812, -0.019604, -0.024115, -0.02502, -0.021293, -0.012392, 0.0016148, 0.019999, 0.041412, 0.064017, 0.085696, 0.1043, 0.11791, 0.1251, 0.1251, 0.11791, 0.1043, 0.085696, 0.064017, 0.041412, 0.019999, 0.0016148, -0.012392, -0.021293, -0.02502, -0.024115, -0.019604, -0.012812, -0.0051582, 0.0020454, 0.0077619, 0.011337, 0.012542, 0.011549, 0.008849, 0.0051294, 0.0011422];
IIR = [0.0084, 0.0252, 0.0252, 0.0084; 1.0000, -2.0727, 1.5292, -0.3892];
x1 = filter(FIR, 1, x);
x2 = filter(IIR(1, :), IIR(2, :), x);
plot(t, x, 'b');
hold on
plot(t, x1, 'r');
plot(t, x2, 'k');
ylim([0.49, 0.64])
xlim([12, 22])
xlabel('t (s)')
legend('x', 'FIR output', 'IIR output')
%%
IIR = [0.0084, 0.0252, 0.0252, 0.0084; 1.0000, -2.0727, 1.5292, -0.3892];
x1 = filter(0.1*ones(1, 10), 1, x);
x2 = filter(IIR(1, :), IIR(2, :), x);
plot(t, x, 'b');
hold on
plot(t, x1, 'r');
plot(t, x2, 'k');
ylim([0.49, 0.64])
xlim([12, 22])
xlabel('t (s)')
legend('x', '10 point mean output', 'IIR output')

%%
IIR = [0.0084, 0.0252, 0.0252, 0.0084; 1.0000, -2.0727, 1.5292, -0.3892];
x1 = filter(0.1*ones(1, 10), 1, x);
x2 = filter(IIR(1, :), IIR(2, :), x1);
plot(t, x, 'b');
hold on
plot(t, x1, 'r');
plot(t, x2, 'k');
ylim([0.49, 0.64])
xlim([12, 22])
xlabel('t (s)')
legend('x', '10 point mean output', 'IIR output of 10 point mean output')

%%
data = load('lg480.mat').data;
t1 = data(:, 1);
x1 = data(:, 2);
data = load('lg720.mat').data;
t2 = data(:, 1);
x2 = data(:, 2);

subplot(1, 2, 1);
plot(t, x);
xlim([11, 22]);
subplot(1, 2, 2);
plot(t2, x2);
xlim([14, 25])