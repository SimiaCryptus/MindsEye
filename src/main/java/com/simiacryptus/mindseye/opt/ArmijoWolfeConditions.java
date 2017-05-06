package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaBuffer;
import com.simiacryptus.mindseye.net.DeltaSet;

import java.util.List;
import java.util.stream.IntStream;

public class ArmijoWolfeConditions implements ScalingStrategy {

    private double c1 = 10e-7;
    private double c2 = 0.9;
    private double alpha = 1.0;
    private double minAlpha = 1e-20;

    @Override
    public Trainable.PointSample step(Trainable subject, DeltaSet direction, Trainable.PointSample measurement, TrainingMonitor monitor) {
        // See http://cs.nyu.edu/overton/mstheses/skajaa/msthesis.pdf page 14
        double mu = 0;
        double nu = Double.POSITIVE_INFINITY;
        double startLineDeriv = dot(measurement.delta.vector(), direction.vector()); // theta'(0)
        double startValue = measurement.value; // theta(0)
        while (true) {
            if (!isAlphaValid()) {
                return measurement;
            }
            if (mu >= nu) {
                return measurement;
            }
            if (Math.abs(alpha) < minAlpha) {
                return measurement;
            }
            final double _alpha = alpha;
            direction.vector().stream().forEach(d -> d.write(_alpha));
            Trainable.PointSample lastSample = subject.measure();
            double thisValue = lastSample.value; // theta(alpha)
            List<DeltaBuffer> newGradient = lastSample.delta.vector();
            double thisLineDeriv = dot(direction.vector(), newGradient); // theta'(alpha)
            if (thisLineDeriv < c2 * startLineDeriv) {
                // Weak Wolfe condition fails
                monitor.log(String.format("WOLFE: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%f <= %f;th'(alpha)=%f < %f", startValue, startLineDeriv, mu, _alpha, nu, thisValue, startValue + alpha * c1 * startLineDeriv, thisLineDeriv, c2 * startLineDeriv));
                mu = alpha;
            } else if (thisValue > startValue + alpha * c1 * startLineDeriv) {
                // Armijo condition fails
                monitor.log(String.format("ARMIJO: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%f > %f;th'(alpha)=%f >= %f", startValue, startLineDeriv, mu, _alpha, nu, thisValue, startValue + alpha * c1 * startLineDeriv, thisLineDeriv, c2 * startLineDeriv));
                nu = alpha;
            } else {
                monitor.log(String.format("END: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%5f;th'(alpha)=%5f", startValue, startLineDeriv, mu, _alpha, nu, thisValue, thisLineDeriv));
                return lastSample;
            }
            direction.vector().stream().forEach(d -> d.write(-_alpha));
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

    private static double dot(List<DeltaBuffer> a, List<DeltaBuffer> b) {
        assert (a.size() == b.size());
        return IntStream.range(0, a.size()).mapToDouble(i -> a.get(i).dot(b.get(i))).sum();
    }

}
