function noisy_signal = add_noise(x_input, variance)
%ADD_NOISE Summary of this function goes here
%   Detailed explanation goes here
noise = variance*randn(size(x_input));
noisy_signal = x_input + noise;
end

