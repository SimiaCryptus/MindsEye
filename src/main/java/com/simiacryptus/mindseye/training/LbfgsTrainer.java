package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.net.DeltaBuffer;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static com.simiacryptus.mindseye.opt.ArrayArrayUtil.*;

public class LbfgsTrainer extends LineSearchTrainer {
  private int minHistory = 2;

  protected List<DeltaBuffer> getDirection(List<DeltaBuffer> startGradient) {
    if(!history.stream().allMatch(x->x.deltaSet.stream().allMatch(y-> Arrays.stream(y).allMatch(d->Double.isFinite(d))))) return startGradient;
    if(!history.stream().allMatch(x->x.targetSet.stream().allMatch(y-> Arrays.stream(y).allMatch(d->Double.isFinite(d))))) return startGradient;
    // See also https://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce
    List<DeltaBuffer> defaultValue = startGradient.stream().map(x -> x.scale(-1)).collect(Collectors.toList());
    List<DeltaBuffer> descent = defaultValue;
    if(history.size() > minHistory) {
      List<double[]> p = descent.stream().map(x -> x.copyDelta()).collect(Collectors.toList());
      assert(p.stream().allMatch(y-> Arrays.stream(y).allMatch(d->Double.isFinite(d))));
      double[] alphas = new double[history.size()];
      for (int i = history.size() - 2; i >= 0; i--) {
        List<double[]> si = minus(history.get(i + 1).targetSet, history.get(i).targetSet);
        List<double[]> yi = minus(history.get(i + 1).deltaSet, history.get(i).deltaSet);
        double denominator = dot(si, yi);
        if(0 == denominator) {
          history.remove(0);
          return getDirection(startGradient);
        }
        alphas[i] = dot(si, p) / denominator;
        p = minus(p, multiply(yi, alphas[i]));
        assert(p.stream().allMatch(y-> Arrays.stream(y).allMatch(d->Double.isFinite(d))));
      }
      List<double[]> sk1 = minus(history.get(history.size() - 1).targetSet, history.get(history.size() - 2).targetSet);
      List<double[]> yk1 = minus(history.get(history.size() - 1).deltaSet, history.get(history.size() - 2).deltaSet);
      p = multiply(p, dot(sk1, yk1) / dot(yk1, yk1));
      assert(p.stream().allMatch(y-> Arrays.stream(y).allMatch(d->Double.isFinite(d))));
      for (int i = 0; i < history.size() - 1; i++) {
        List<double[]> si = minus(history.get(i + 1).targetSet, history.get(i).targetSet);
        List<double[]> yi = minus(history.get(i + 1).deltaSet, history.get(i).deltaSet);
        double beta = dot(yi, p) / dot(si, yi);
        p = add(p, multiply(si, alphas[i] - beta));
        assert(p.stream().allMatch(y-> Arrays.stream(y).allMatch(d->Double.isFinite(d))));
      }
      List<double[]> _p = p;
      for (int i = 0; i < descent.size(); i++) {
        int _i = i;
        Arrays.setAll(descent.get(i).delta, j -> _p.get(_i)[j]);
      }
    }
    return descent;
  }


  public int getMinHistory() {
    return minHistory;
  }

  public LbfgsTrainer setMinHistory(int minHistory) {
    this.minHistory = minHistory;
    return this;
  }
}
