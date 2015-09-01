package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

public class ChampionTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(ChampionTrainer.class);

  private GradientDescentTrainer best = null;
  private GradientDescentTrainer current = null;
  private boolean verbose = false;
  
  public ChampionTrainer() {
    this(new GradientDescentTrainer());
  }
  
  public ChampionTrainer(final GradientDescentTrainer current) {
    assert null != current;
    setCurrent(current);
  }

  public GradientDescentTrainer getBest() {
    return this.best;
  }
  
  public GradientDescentTrainer getCurrent() {
    return this.current;
  }

  public List<NNLayer> getLayers() {
    return getCurrent().getLayers();
  }
  
  public PipelineNetwork getNetwork() {
    return this.current.getNetwork();
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  public void revert(TrainingContext trainingContext) {
    final GradientDescentTrainer best = getBest();
    if (null != best)
    {
      if (isVerbose()) {
        ChampionTrainer.log.debug(String.format("Revert to best = %s", null == best ? null : best.error(trainingContext)));
        // log.debug(String.format("Discarding %s", best.getFirst().get(0).getNet()));
      }
      setCurrent(Util.kryo().copy(best));
    }
  }
  
  public ChampionTrainer setBest(final GradientDescentTrainer best) {
    this.best = best;
    return this;
  }
  
  public ChampionTrainer setCurrent(final GradientDescentTrainer current) {
    this.current = current;
    return this;
  }
  
  public ChampionTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    getCurrent().setVerbose(verbose);
    return this;
  }
  
  public Double step(TrainingContext trainingContext, final double[] rates) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    getCurrent().trainSet(trainingContext, rates);
    updateBest(trainingContext);
    trainingContext.gradientSteps.increment();
    if (this.verbose)
    {
      ChampionTrainer.log.debug(String.format("Trained Error: %s (%s) with rate %s in %.03fs",
          getCurrent().error(trainingContext), Arrays.toString(getCurrent().getError()), getCurrent().getRate(),
          (System.currentTimeMillis() - startMs) / 1000.));
    }
    return getCurrent().error(trainingContext);
  }
  
  protected void updateBest(TrainingContext trainingContext) {
    final GradientDescentTrainer best = getBest();
    GradientDescentTrainer current = getCurrent();
    double currentError = current.error(trainingContext);
    double bestError = null==best?Double.POSITIVE_INFINITY:best.error(trainingContext);
    if (Double.isFinite(currentError) && (null == best || bestError > currentError)) {
      if (isVerbose()) {
        ChampionTrainer.log.debug(String.format("New best Error %s > %s", currentError, null == best ? "null" : bestError));
        // log.debug(String.format("Best: %s", currentNetworks.get(0).getNet()));
      }
      setBest(Util.kryo().copy(current));
    }
  }
  
}
