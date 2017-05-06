package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaBuffer;
import com.simiacryptus.mindseye.net.DeltaSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static com.simiacryptus.mindseye.opt.ArrayArrayUtil.add;
import static com.simiacryptus.mindseye.opt.ArrayArrayUtil.dot;

public class LBFGS implements OrientationStrategy {
    public final ArrayList<Trainable.PointSample> history = new ArrayList<>();
    private int minHistory = 3;

    @Override
    public DeltaSet orient(Trainable.PointSample measurement, TrainingMonitor monitor) {
        DeltaSet startGradient = measurement.delta;
        if (!history.stream().allMatch(x -> x.delta.vector().stream().allMatch(y -> Arrays.stream(y.delta).allMatch(d -> Double.isFinite(d)))))
            return startGradient;
        if (!history.stream().allMatch(x -> x.weights.vector().stream().allMatch(y -> Arrays.stream(y.delta).allMatch(d -> Double.isFinite(d)))))
            return startGradient;
        // See also https://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce
        List<DeltaBuffer> defaultValue = startGradient.vector().stream().map(x -> x.scale(-1)).collect(Collectors.toList());
        List<DeltaBuffer> descent = defaultValue;
        if (history.size() > minHistory) {
            List<double[]> p = descent.stream().map(x -> x.copyDelta()).collect(Collectors.toList());
            assert (p.stream().allMatch(y -> Arrays.stream(y).allMatch(d -> Double.isFinite(d))));
            double[] alphas = new double[history.size()];
            for (int i = history.size() - 2; i >= 0; i--) {
                List<double[]> si = minus((history.get(i + 1).weights.vector()), history.get(i).weights.vector());
                List<double[]> yi = minus(history.get(i + 1).delta.vector(), history.get(i).delta.vector());
                double denominator = dot(si, yi);
                if (0 == denominator) {
                    history.remove(0);
                    return orient(measurement, monitor);
                }
                alphas[i] = dot(si, p) / denominator;
                p = ArrayArrayUtil.minus(p, ArrayArrayUtil.multiply(yi, alphas[i]));
                assert (p.stream().allMatch(y -> Arrays.stream(y).allMatch(d -> Double.isFinite(d))));
            }
            List<double[]> sk1 = minus(history.get(history.size() - 1).weights.vector(), history.get(history.size() - 2).weights.vector());
            List<double[]> yk1 = minus(history.get(history.size() - 1).delta.vector(), history.get(history.size() - 2).delta.vector());
            p = ArrayArrayUtil.multiply(p, dot(sk1, yk1) / dot(yk1, yk1));
            assert (p.stream().allMatch(y -> Arrays.stream(y).allMatch(d -> Double.isFinite(d))));
            for (int i = 0; i < history.size() - 1; i++) {
                List<double[]> si = minus(history.get(i + 1).weights.vector(), history.get(i).weights.vector());
                List<double[]> yi = minus(history.get(i + 1).delta.vector(), history.get(i).delta.vector());
                double beta = dot(yi, p) / dot(si, yi);
                p = add(p, ArrayArrayUtil.multiply(si, alphas[i] - beta));
                assert (p.stream().allMatch(y -> Arrays.stream(y).allMatch(d -> Double.isFinite(d))));
            }
            List<double[]> _p = p;
            for (int i = 0; i < descent.size(); i++) {
                int _i = i;
                Arrays.setAll(descent.get(i).delta, j -> _p.get(_i)[j]);
            }
        }
        if (accept(measurement.delta.vector(), descent)) {
            history.remove(0);
            return orient(measurement, monitor);
        }
        return DeltaSet.fromList(descent);
    }

    private boolean accept(List<DeltaBuffer> gradient, List<DeltaBuffer> direction) {
        return dot(cvt(gradient), cvt(direction)) > 0;
    }

    private List<double[]> minus(List<DeltaBuffer> a, List<DeltaBuffer> b) {
        return ArrayArrayUtil.minus(cvt(a), cvt(b));
    }

    private List<double[]> cvt(List<DeltaBuffer> vector) {
        return vector.stream().map(x -> x.delta).collect(Collectors.toList());
    }
}
