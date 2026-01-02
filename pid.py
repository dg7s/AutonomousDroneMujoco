class PID:
    def __init__(
            self, gain_prop: float, gain_int: float, gain_der: float, sensor_period: float,
            output_limits: tuple[float, float]
            ):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: define additional attributes you might need
        self.integral_sum = 0
        self.output_limits = output_limits
        # END OF TODO


    # TODO: implement function which computes the output signal
    # The controller should output only in the range of output_limits
    def output_signal(self, commanded_variable: float, sensor_readings: list[float]) -> float:
        current_val = sensor_readings[0]
        previous_val = sensor_readings[1]

        error = commanded_variable - current_val
        P = self.gain_prop * error

        self.integral_sum += error * self.sensor_period
        I = self.gain_int * self.integral_sum

        D = -self.gain_der * (current_val - previous_val) / self.sensor_period

        output_signal = P + I + D
        output_signal = max(output_signal, self.output_limits[0])
        output_signal = min(output_signal, self.output_limits[1])
        return output_signal
    # END OF TODO
