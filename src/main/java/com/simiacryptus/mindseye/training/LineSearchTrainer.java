package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.net.DeltaBuffer;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleFunction;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LineSearchTrainer extends StochasticTrainer {

  private static final Logger log = LoggerFactory.getLogger(LineSearchTrainer.class);

  private double alpha = 1.0;
  private double linesearch_c1 = 10e-7;
  private double linesearch_c2 = 0.9;
  protected final List<ExecutionRecord> history = new ArrayList<>();
  private int memorySize = 10;
  private double minAlpha = 1e-20;

  public static class ExecutionRecord {
    final List<double[]> deltaSet;
    final List<double[]> targetSet;

    public ExecutionRecord(NNResult evalResult) {
      DeltaSet deltaSet = new DeltaSet();
      evalResult.accumulate(deltaSet);
      this.deltaSet = deltaSet.vector().stream().map(x->x.copyDelta()).collect(Collectors.toList());
      this.targetSet = deltaSet.vector().stream().map(x->x.copyTarget()).collect(Collectors.toList());

    }
  }

  @Override
  public TrainingStep step(final TrainingContext trainingContext) throws TerminationCondition {
    //final long startMs = System.currentTimeMillis();
    double mu;
    double startValue = Double.NaN;
    double thisValue = Double.NaN;
    double nu;
    List<DeltaBuffer> direction;
    Tensor[][] data;
    double startLineDeriv;
    do {
      updateHash();
      data = selectTrainingData();
      if (data.length == 0) {
        return new TrainingStep(Double.NaN, Double.NaN, false);
      }
      assert 0 < data.length;
      NNResult evalResult = getPrimaryNode().get(createBatchExeContext(trainingContext, data));
      history.add(new ExecutionRecord(evalResult));
      List<DeltaBuffer> startGradient = getDelta(evalResult);
      direction = getDirection(startGradient);
      if(history.size() > memorySize) history.remove(0);
      // See http://cs.nyu.edu/overton/mstheses/skajaa/msthesis.pdf page 14
      mu = 0;
      nu = Double.POSITIVE_INFINITY;
      startLineDeriv = dot(direction, startGradient); // theta'(0)
      if(!Double.isFinite(startLineDeriv) || 0 <= startLineDeriv) {
        System.err.println(String.format("CONTIUNUED due to startDeriv=%s", startLineDeriv));
        history.remove(0);
        continue;
      }
      startValue = summarize(evalResult).rms; // theta(0)
      if(!Double.isFinite(startValue)) {
        System.err.println(String.format("CONTIUNUED due to startValue=%s", startValue));
        history.remove(0);
        continue;
      }
      if(!isAlphaValid()) alpha = 1.0;
      thisValue = Double.NaN;
    } while(false);

    while(isAlphaValid() && mu < nu && Math.abs(alpha) > minAlpha) {
      double _alpha = alpha;
      direction.stream().forEach(d -> d.write(_alpha));
      NNResult validationResult = eval(trainingContext, data);
      thisValue = summarize(validationResult).rms; // theta(alpha)
      List<DeltaBuffer> newGradient = getDelta(validationResult);
      double thisLineDeriv = dot(direction, newGradient); // theta'(alpha)
      if(thisLineDeriv < linesearch_c2 * startLineDeriv) {
        // Weak Wolfe condition fails
        if(isVerbose()) System.err.println(String.format("WOLFE: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%f <= %f;th'(alpha)=%f < %f", startValue, startLineDeriv, mu, _alpha, nu, thisValue, startValue + alpha * linesearch_c1 * startLineDeriv, thisLineDeriv, linesearch_c2 * startLineDeriv));
        mu = alpha;
      } else if(thisValue > startValue + alpha * linesearch_c1 * startLineDeriv) {
        // Armijo condition fails
        if(isVerbose()) System.err.println(String.format("ARMIJO: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%f > %f;th'(alpha)=%f >= %f", startValue, startLineDeriv, mu, _alpha, nu, thisValue, startValue + alpha * linesearch_c1 * startLineDeriv, thisLineDeriv, linesearch_c2 * startLineDeriv));
        nu = alpha;
      } else{
        if(isVerbose()) System.err.println(String.format("END: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%5f;th'(alpha)=%5f", startValue, startLineDeriv, mu, _alpha, nu, thisValue, thisLineDeriv));
        break;
      }
      direction.stream().forEach(d -> d.write(-_alpha));
      if(Double.isFinite(nu)) {
        alpha = (mu + nu) / 2;
      } else {
        alpha = 2 * alpha;
      }
    }
    return new TrainingStep(startValue, thisValue, true);
  }

  protected List<DeltaBuffer> getDirection(List<DeltaBuffer> startGradient) {
    return startGradient.stream().map(x->x.scale(-1.0)).collect(Collectors.toList());
  }

  private boolean isAlphaValid() {
    return Double.isFinite(alpha) && (0 <= alpha);
  }

  private double dot(List<DeltaBuffer> a, List<DeltaBuffer> b) {
    assert(a.size()==b.size());
    return IntStream.range(0, a.size()).mapToDouble(i->a.get(i).dot(b.get(i))).sum();
  }

  private List<DeltaBuffer> getDelta(NNResult evalResult) {
    final DeltaSet delta = new DeltaSet();
    evalResult.accumulate(delta);
    return delta.vector();
  }

  @Override
  public void reset() {
    setError(Double.NaN);
  }

  public double getLinesearch_c1() {
    return linesearch_c1;
  }

  public LineSearchTrainer setLinesearch_c1(double linesearch_c1) {
    this.linesearch_c1 = linesearch_c1;
    return this;
  }

  public double getLinesearch_c2() {
    return linesearch_c2;
  }

  public LineSearchTrainer setLinesearch_c2(double linesearch_c2) {
    this.linesearch_c2 = linesearch_c2;
    return this;
  }

  public int getMemorySize() {
    return memorySize;
  }

  public LineSearchTrainer setMemorySize(int memorySize) {
    this.memorySize = memorySize;
    return this;
  }

}
