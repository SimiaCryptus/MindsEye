package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.util.Util;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.concurrent.TimeUnit;

public class IterativeTrainer {

    public static class Step {
        public final Trainable.PointSample point;
        public final long time = System.currentTimeMillis();
        public final long iteration;

        private Step(Trainable.PointSample point, long iteration) {
            this.point = point;
            this.iteration = iteration;
        }
    }

    private final Trainable subject;
    private Duration timeout;
    private double terminateThreshold;
    private OrientationStrategy orientation = new LBFGS();
    private ScalingStrategy scaling = new ArmijoWolfeConditions();
    private TrainingMonitor monitor = new TrainingMonitor();
    private int currentIteration = 0;

    public IterativeTrainer(Trainable subject) {
        this.subject = subject;
        timeout = Duration.of(5, ChronoUnit.MINUTES);
        terminateThreshold = Double.NEGATIVE_INFINITY;
    }

    public double run() {
        long timeoutMs = System.currentTimeMillis() + timeout.toMillis();
        Trainable.PointSample currentPoint = measure();
        while(timeoutMs > System.currentTimeMillis() && currentPoint.value > terminateThreshold) {
            System.gc();
            currentPoint = measure();
            DeltaSet direction = orientation.orient(currentPoint, monitor);
            currentPoint = scaling.step(subject, direction, currentPoint, monitor);
            monitor.log(String.format("Iteration %s complete. Error: %s", currentIteration, currentPoint.value));
            monitor.onStepComplete(new Step(currentPoint, currentIteration++));
        }
        return null==currentPoint?Double.NaN:currentPoint.value;
    }

    public Trainable.PointSample measure() {
        Trainable.PointSample currentPoint;
        int retries = 0;
        do {
            if(3 < retries++) throw new RuntimeException();
            subject.resetSampling();
            currentPoint = subject.measure();
        } while(!Double.isFinite(currentPoint.value));
        assert(Double.isFinite(currentPoint.value));
        return currentPoint;
    }

    public Duration getTimeout() {
        return timeout;
    }

    public IterativeTrainer setTimeout(int number, TimeUnit units) {
        return setTimeout(number, Util.cvt(units));
    }

    public IterativeTrainer setTimeout(int number, TemporalUnit units) {
        this.timeout = Duration.of(number, units);
        return this;
    }

    public IterativeTrainer setTimeout(Duration timeout) {
        this.timeout = timeout;
        return this;
    }

    public double getTerminateThreshold() {
        return terminateThreshold;
    }

    public IterativeTrainer setTerminateThreshold(double terminateThreshold) {
        this.terminateThreshold = terminateThreshold;
        return this;
    }

    public OrientationStrategy getOrientation() {
        return orientation;
    }

    public IterativeTrainer setOrientation(OrientationStrategy orientation) {
        this.orientation = orientation;
        return this;
    }

    public ScalingStrategy getScaling() {
        return scaling;
    }

    public IterativeTrainer setScaling(ScalingStrategy scaling) {
        this.scaling = scaling;
        return this;
    }

    public TrainingMonitor getMonitor() {
        return monitor;
    }

    public IterativeTrainer setMonitor(TrainingMonitor monitor) {
        this.monitor = monitor;
        return this;
    }
}
