package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaBuffer;
import com.simiacryptus.mindseye.net.DeltaSet;

import java.util.List;
import java.util.stream.IntStream;

public class ArmijoWolfeConditions implements LineSearchStrategy {

    private double c1 = 1e-6;
    private double c2 = 0.9;
    private double alpha = 1.0;
    private double minAlpha = 1e-20;
    private double alphaGrowth = Math.pow(3.0,Math.pow(5.0,-1.0));

    @Override
    public Trainable.PointSample step(LineSearchCursor cursor, TrainingMonitor monitor) {
        alpha *= alphaGrowth; // Keep memory of alpha from one iteration to next, but have a bias for growing the value
        // See http://cs.nyu.edu/overton/mstheses/skajaa/msthesis.pdf page 14
        double mu = 0;
        double nu = Double.POSITIVE_INFINITY;
        LineSearchPoint startPoint = cursor.step(0, monitor);
        double startLineDeriv = startPoint.derivative; // theta'(0)
        double startValue = startPoint.point.value; // theta(0)
        while (true) {
            if (!isAlphaValid()) {
                monitor.log(String.format("INVALID ALPHA: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s", startValue, startLineDeriv, mu, alpha, nu));
                return cursor.step(0, monitor).point;
            }
            if (mu >= nu) {
                monitor.log(String.format("mu >= nu: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s", startValue, startLineDeriv, mu, alpha, nu));
                return cursor.step(0, monitor).point;
            }
            if ((nu / mu) < (11.0/10.0)) {
                monitor.log(String.format("mu >= nu: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s", startValue, startLineDeriv, mu, alpha, nu));
                return cursor.step(0, monitor).point;
            }
            if (Math.abs(alpha) < minAlpha) {
                monitor.log(String.format("MIN ALPHA: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s", startValue, startLineDeriv, mu, alpha, nu));
                return cursor.step(0, monitor).point;
            }
            LineSearchPoint lastStep = cursor.step(alpha, monitor);
            if (lastStep.derivative < c2 * startLineDeriv) {
                // Weak Wolfe condition fails
                monitor.log(String.format("WOLFE: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%f <= %f;th'(alpha)=%f < %f",
                        startValue, startLineDeriv, mu, alpha, nu, lastStep.point.value, startValue + alpha * c1 * startLineDeriv, lastStep.derivative, c2 * startLineDeriv));
                mu = alpha;
            } else if (lastStep.point.value > startValue + alpha * c1 * startLineDeriv) {
                // Armijo condition fails
                monitor.log(String.format("ARMIJO: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%f > %f;th'(alpha)=%f >= %f",
                        startValue, startLineDeriv, mu, alpha, nu, lastStep.point.value, startValue + alpha * c1 * startLineDeriv, lastStep.derivative, c2 * startLineDeriv));
                nu = alpha;
            } else {
                monitor.log(String.format("END: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%5f;th'(alpha)=%5f",
                        startValue, startLineDeriv, mu, alpha, nu, lastStep.point.value, lastStep.derivative));
                return lastStep.point;
            }
            if (Double.isFinite(nu)) {
                alpha = (mu + nu) / 2;
            } else {
                alpha = 2 * alpha;
            }
        }
    }

    private boolean isAlphaValid() {
        return Double.isFinite(alpha) && (0 <= alpha);
    }

    public double getAlphaGrowth() {
        return alphaGrowth;
    }

    public ArmijoWolfeConditions setAlphaGrowth(double alphaGrowth) {
        this.alphaGrowth = alphaGrowth;
        return this;
    }

    public double getC1() {
        return c1;
    }

    public ArmijoWolfeConditions setC1(double c1) {
        this.c1 = c1;
        return this;
    }

    public double getC2() {
        return c2;
    }

    public ArmijoWolfeConditions setC2(double c2) {
        this.c2 = c2;
        return this;
    }

    public double getAlpha() {
        return alpha;
    }

    public ArmijoWolfeConditions setAlpha(double alpha) {
        this.alpha = alpha;
        return this;
    }
}
