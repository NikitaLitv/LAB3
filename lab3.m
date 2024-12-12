clc;
close all;

% Input and target output generation
x = 0.1:1/22:1;
x_test = 0.1:1/200:1;
y = ((1 + 0.6 * sin(2 * pi * x / 0.7)) + 0.3 * sin(2 * pi * x)) / 2;
y_test = ((1 + 0.6 * sin(2 * pi * x_test / 0.7)) + 0.3 * sin(2 * pi * x_test)) / 2;

figure(1);
plot(x_test, y_test, 'b', 'LineWidth', 1.5);
grid on;
hold on;

% Initialize RBF parameters
%c1 = 0.2; 
%c2 = 0.9; 
%r1 = 0.15;
%r2 = 0.15;

c1 = rand();
c2 = rand();
r1 = rand();
r2 = rand();

% Gaussian RBF functions
F1 = exp(-((x - c1).^2) / (2 * r1^2));
F2 = exp(-((x - c2).^2) / (2 * r2^2));

% Initialize output layer weights (w1, w2, w0)
w1 = rand();
w2 = rand();
w0 = rand();

% Learning rate
eta = 0.01;
eta_c_r=0.001;

%Number of learning cycles
num_epochs = 10000;

%Errors
err=0;
sqr_err=0;
ms_err=0;
prev_err=zeros(1, num_epochs);

% Training loop with gradient descent for RBF parameters
for epoch = 1:num_epochs
    sqr_err=0;
    for i = 1:length(x)
        % Calculate RBF outputs for current input
        phi1 = exp(-((x(i) - c1)^2) / (2 * r1^2));
        phi2 = exp(-((x(i) - c2)^2) / (2 * r2^2));

        % Compute network output
        y_pred(i) = w1 * phi1 + w2 * phi2 + w0; % Store y_pred for each x(i)

        % Error
        err = y(i) - y_pred(i);
        sqr_err=sqr_err+err^2;

        % Update weights
        w1 = w1 + eta * err * phi1;
        w2 = w2 + eta * err * phi2;
        w0 = w0 + eta * err;

        % Update RBF parameters
        c1 = c1 + eta_c_r * err * w1 * phi1 * ((x(i) - c1) / r1^2);
        c2 = c2 + eta_c_r * err * w2 * phi2 * ((x(i) - c2) / r2^2);
        r1 = r1 + eta_c_r * err * w1 * phi1 * ((x(i) - c1)^2 / r1^3);
        r2 = r2 + eta_c_r * err * w2 * phi2 * ((x(i) - c2)^2 / r2^3);

        
    end
    ms_err=sqr_err/length(x_test);
    prev_err(i)=ms_err;
end

% Testing the RBF network
Y_test = zeros(1, length(x_test));
for i = 1:length(x_test)
    phi1_test = exp(-((x_test(i) - c1)^2) / (2 * r1^2));
    phi2_test = exp(-((x_test(i) - c2)^2) / (2 * r2^2));

    % Compute network output
    Y_test(i) = w1 * phi1_test + w2 * phi2_test + w0;
end

% Plot results
plot(x_test, Y_test, 'r', 'LineWidth', 1.5);
hold off;
legend('Target', 'Predicted');
title('RBF Network Approximation');
xlabel('x');
ylabel('y');

figure(2)
plot(x, y, 'b', 'LineWidth', 1.5);
grid on;
hold on;
plot(x, y_pred, 'r', 'LineWidth', 1.5);
legend('Target', 'Predicted');
title('RBF Network Training Approximation');
xlabel('x');
ylabel('y');
hold off

%figure(3)
%plot(1:num_epochs, prev_err, 'b', 'LineWidth', 1.5);
%grid on;
%title('MSE');
%xlabel('Epochs');
%ylabel('MSE');