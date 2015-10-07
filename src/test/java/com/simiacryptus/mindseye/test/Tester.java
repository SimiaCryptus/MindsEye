package com.simiacryptus.mindseye.test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.training.DevelopmentTrainer;
import com.simiacryptus.mindseye.training.DynamicRateTrainer;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;
import com.simiacryptus.mindseye.training.NetInitializer;
import com.simiacryptus.mindseye.training.TrainingComponent;
import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

/**
 * Encapsulates overall network architecture, training method and data.
 *
 * @author Andrew Charneski
 */
public class Tester {

  static final Logger log = LoggerFactory.getLogger(Tester.class);

  public static DAGNetwork initPredictionNetwork(final NNLayer<?> predictor, final NNLayer<?> loss) {
    final DAGNetwork dagNetwork = new DAGNetwork();
    dagNetwork.add(predictor);
    dagNetwork.addLossComponent(loss);
    return dagNetwork;
  }

  public final List<BiFunction<DAGNetwork, TrainingContext, Void>> handler = new ArrayList<>();

  protected GradientDescentTrainer gradientTrainer;
  protected DynamicRateTrainer dynamicTrainer;
  protected DevelopmentTrainer devtrainer;
  private boolean parallel = true;
  private final TrainingContext trainingContext;


  public Tester() {
    super();
    initLayers();
    trainingContext = new TrainingContext().setTimeout(1, TimeUnit.MINUTES);
  }

  public void initLayers() {
    gradientTrainer = new GradientDescentTrainer();
    dynamicTrainer = new DynamicRateTrainer(gradientTrainer);
    devtrainer = new DevelopmentTrainer(dynamicTrainer);
  }

  public Tester init(final NDArray[][] samples, final NNLayer<DAGNetwork> pipelineNetwork, final NNLayer<?> lossLayer) {
    DAGNetwork initPredictionNetwork = initPredictionNetwork(pipelineNetwork, lossLayer);
    //new NetInitializer().initialize(initPredictionNetwork);
    gradientTrainer.setNet(initPredictionNetwork);
    gradientTrainer.setData(samples);
    return this;
  }

  public boolean isParallel() {
    return this.parallel;
  }

  public Tester setDynamicRate(final double d) {
    getGradientDescentTrainer().setRate(d);
    return this;
  }

  public Tester setMaxDynamicRate(final double d) {
    getDynamicRateTrainer().setMaxRate(d);
    return this;
  }

  public Tester setMinDynamicRate(final double d) {
    getDynamicRateTrainer().setMinRate(d);
    return this;
  }

  public Tester setParallel(final boolean parallel) {
    this.parallel = parallel;
    return this;
  }

  public Tester setStaticRate(final double d) {
    getGradientDescentTrainer().setRate(d);
    return this;
  }

  public GradientDescentTrainer getGradientDescentTrainer() {
    return gradientTrainer;
  }

  public DynamicRateTrainer getDynamicRateTrainer() {
    return dynamicTrainer;
  }

  public Tester setVerbose(final boolean b) {
    getGradientDescentTrainer().setVerbose(b);
    getDynamicRateTrainer().setVerbose(b);
    getDevtrainer().setVerbose(b);
    return this;
  }

  public void train(final double stopError, final TrainingContext trainingContext) throws TerminationCondition {
    trainingContext.terminalErr = stopError;
    getDevtrainer().step(trainingContext);
  }

  public TrainingContext trainingContext() {
    return this.trainingContext;
  }

  public long verifyConvergence(final double convergence, final int reps) {
    return verifyConvergence(convergence, reps, reps);
  }

  public long verifyConvergence(final double convergence, final int reps, final int minSuccess) {
    IntStream range = IntStream.range(0, reps);
    if (isParallel()) {
      range = range.parallel();
    }
    final long succeesses = range.filter(i -> {
      final TrainingComponent trainerCpy = Util.kryo().copy(getDevtrainer());
      final TrainingContext contextCpy = Util.kryo().copy(trainingContext());
      getInitializer().initialize(trainerCpy.getNet());
      //contextCpy.setTimeout(1, TimeUnit.MINUTES);
      boolean hasConverged = false;
      try {
        contextCpy.terminalErr = convergence;
        final double error = trainerCpy.step(contextCpy).finalError();
        final DAGNetwork net = trainerCpy.getNet();
        this.handler.stream().forEach(h -> h.apply(net, contextCpy));
        hasConverged = error <= convergence;
        if (!hasConverged) {
          Tester.log.debug(String.format("Not Converged: %s <= %s", error, convergence));
        }
      } catch (final Throwable e) {
        Tester.log.debug("Not Converged", e);
      }
      return hasConverged;
    }).count();
    if (minSuccess > succeesses)
      throw new RuntimeException(String.format("%s out of %s converged", succeesses, reps));
    return succeesses;
  }

  public NetInitializer getInitializer() {
    return new NetInitializer();
  }

  public DevelopmentTrainer getDevtrainer() {
    return devtrainer;
  }

  public DAGNetwork getNet() {
    return getDevtrainer().getNet();
  }

}
