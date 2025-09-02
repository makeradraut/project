% DC Motor Parameters
Ra = 1;       % Armature resistance (Ohm)
La = 0.5;     % Armature inductance (H)
Ke = 2.7;    % Back EMF constant (V.s/rad)
Kt = 2.7;    % Torque constant (Nm/A)
J = 0.01;     % Rotor inertia (kg.m^2)
B = 0.1;      % Viscous damping coefficient (Nm.s)

% Input voltage
Va = 120;      % Constant input voltage (V)
% State-Space Matrices
A = [-Ra/La, -Ke/La;
      Kt/J,  -B/J];

B = [1/La; 0];
C = [0 1];  % Output angular speed
D = 0;
%%
sys = ss(A, B, C, D);
% Time Vector
t = 0:0.001:2;

% Input Voltage
u = Va * ones(size(t));

% Simulate using lsim
initial_state = [0; 0];  % [ia(0); omega(0)]
[y, t, x] = lsim(sys, u, t, initial_state);
%%
% Plot results
figure;
subplot(2,1,1);
plot(t, x(:,1), 'b', 'LineWidth', 2);
ylabel('Armature Current (A)');
grid on;

subplot(2,1,2);
plot(t, x(:,2), 'r', 'LineWidth', 2);
ylabel('Angular Speed (rad/s)');
xlabel('Time (s)');
grid on;
sgtitle('DC Motor Response');
%%


% Parameters
R = 1;              % Ohm
L0 = 0.01;          % H
L1 = 0.04;          % H
Nr = 4;             % Rotor pole pairs
omega = 100;        % rad/s (mechanical)
Vd = 48;            % Applied d-axis voltage
%%
% Time vector
t = 0:1e-4:0.1;

% Initialize
id = zeros(size(t));
theta = omega * t;  % Rotor position
Ld = @(theta) L0 + L1 * cos(Nr * theta);
dLd_dtheta = @(theta) -Nr * L1 * sin(Nr * theta);

% Numerical integration
for k = 2:length(t)
    dt = t(k) - t(k-1);
    L_now = Ld(theta(k-1));
    dLdt = dLd_dtheta(theta(k-1)) * omega;
    did_dt = (Vd - R * id(k-1) - dLdt * id(k-1)) / L_now;
    id(k) = id(k-1) + did_dt * dt;
end

% Torque
Torque = 0.5 .* id.^2 .* dLd_dtheta(theta);
%%
% Plot
figure;
subplot(2,1,1);
plot(t, id, 'b');
ylabel('i_d (A)');
grid on;

subplot(2,1,2);
plot(t, Torque, 'r');
ylabel('Torque (Nm)');
xlabel('Time (s)');
grid on;
