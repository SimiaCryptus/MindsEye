package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.EntropyLossLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

/**
 * Encapsulates overall network architecture, training method and data.
 *
 * @author Andrew Charneski
 */
public class Tester {

  // private static final class PhasedLossLayer extends WrapperLayer {
  // private PhasedLossLayer() {
  //// super(new MaxEntropyLossLayer());
  // super(new EntropyLossLayer());
  //// super(new SqLossLayer());
  // }
  //
  // @Override
  // public NNLayer<?> evolve() {
  // return null;
  //// if(getInner() instanceof EntropyLossLayer) return null;
  //// setInner(new EntropyLossLayer());
  //// return this;
  // }
  // }

  static final Logger log = LoggerFactory.getLogger(Tester.class);

  public static DAGNetwork initPredictionNetwork(final NNLayer<?> predictor, final NNLayer<?> loss) {
    final DAGNetwork dagNetwork = new DAGNetwork();
    dagNetwork.add(predictor);
    dagNetwork.add2(loss);
    return dagNetwork;
  }

  public static boolean test(final PopulationTrainer self, final double convergence, final TrainingContext trainingContext,
      final List<BiFunction<DAGNetwork, TrainingContext, Void>> handler) {
    boolean hasConverged = false;
    try {
      self.getDynamicRateTrainer().setStopError(convergence);
      final double error = self.step(trainingContext);
      final DAGNetwork net = self.getNet();
      handler.stream().forEach(h -> h.apply(net, trainingContext));
      hasConverged = error <= convergence;
      if (!hasConverged) {
        log.debug(String.format("Not Converged: %s <= %s", error, convergence));
      }
    } catch (final Throwable e) {
      log.debug("Not Converged", e);
    }
    return hasConverged;
  }

  public final List<BiFunction<DAGNetwork, TrainingContext, Void>> handler = new ArrayList<>();

  private PopulationTrainer inner = new PopulationTrainer();

  private boolean parallel = true;

  TrainingContext trainingContext = new TrainingContext();

  public PopulationTrainer getInner() {
    return this.inner;
  }

  /**
   * @deprecated Use {@link #init(NDArray[][],DAGNetwork,NNLayer<?>)} instead
   */
  @Deprecated
  public Tester init(final DAGNetwork pipelineNetwork, final NDArray[][] samples, final NNLayer<?> lossLayer) {
    return init(samples, pipelineNetwork, lossLayer);
  }

  public Tester init(final NDArray[][] samples, final DAGNetwork univariateNetwork) {
    final GradientDescentTrainer gradientDescentTrainer = getInner().getDynamicRateTrainer().getGradientDescentTrainer();
    gradientDescentTrainer.setNet(univariateNetwork);
    gradientDescentTrainer.setMasterTrainingData(samples);
    return this;
  }

  public Tester init(final NDArray[][] samples, final DAGNetwork pipelineNetwork, final NNLayer<?> lossLayer) {
    return init(samples, initPredictionNetwork(pipelineNetwork, lossLayer));
  }

  /**
   * @deprecated Use {@link #init(NDArray[][],DAGNetwork,EntropyLossLayer)}
   *             instead
   */
  @Deprecated
  public Tester initEntropy(final NDArray[][] samples, final DAGNetwork pipelineNetwork) {
    return init(samples, pipelineNetwork, new EntropyLossLayer());
  }

  public boolean isParallel() {
    return this.parallel;
  }

  public Tester setDynamicRate(final double d) {
    getInner().getDynamicRateTrainer().getGradientDescentTrainer().setRate(d);
    return this;
  }

  public void setInner(final PopulationTrainer inner) {
    this.inner = inner;
  }

  public Tester setMaxDynamicRate(final double d) {
    getInner().getDynamicRateTrainer().setMaxRate(d);
    return this;
  }

  public Tester setMinDynamicRate(final double d) {
    getInner().getDynamicRateTrainer().setMinRate(d);
    return this;
  }

  public Tester setMutationAmplitude(final double d) {
    getInner().setAmplitude(d);
    return this;
  }

  public Tester setParallel(final boolean parallel) {
    this.parallel = parallel;
    return this;
  }

  public Tester setStaticRate(final double d) {
    getInner().getDynamicRateTrainer().getGradientDescentTrainer().setRate(d);
    return this;
  }

  public Tester setVerbose(final boolean b) {
    getInner().setVerbose(b);
    return this;
  }

  public void train(final double stopError, final TrainingContext trainingContext) throws TerminationCondition {
    getInner().getDynamicRateTrainer().setStopError(stopError);
    getInner().step(trainingContext);
  }

  private TrainingContext trainingContext() {
    return this.trainingContext;
  }

  public long verifyConvergence(final double convergence, final int reps) {
    return verifyConvergence(convergence, reps, reps);
  }

  public long verifyConvergence(final double convergence, final int reps, final int minSuccess) {
    IntStream range = IntStream.range(0, reps);
    if (isParallel()) {
      //range = range.parallel();
    }
    final long succeesses = range.filter(i -> {
      final PopulationTrainer trainerCpy = Util.kryo().copy(getInner());
      final TrainingContext contextCpy = Util.kryo().copy(trainingContext());
      contextCpy.setTimeout(1, TimeUnit.MINUTES);
      return Tester.test(trainerCpy, convergence, contextCpy, this.handler);
    }).count();
    if (minSuccess > succeesses)
      throw new RuntimeException(String.format("%s out of %s converged", succeesses, reps));
    return succeesses;
  }

}
