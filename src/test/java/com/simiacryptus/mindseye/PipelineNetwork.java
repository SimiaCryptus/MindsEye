package com.simiacryptus.mindseye;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.simiacryptus.mindseye.layers.DenseSynapseLayer;

public class PipelineNetwork extends NNLayer {
  private List<NNLayer> layers = new ArrayList<NNLayer>();
  private double quantum = 0.001;
  private double lastRms = Double.MAX_VALUE;
  
  @Override
  public NNResult eval(NNResult array) {
    NNResult r = array;
    for (NNLayer l : layers)
      r = l.eval(r);
    return r;
  }
  
  public PipelineNetwork add(NNLayer layer) {
    layers.add(layer);
    return this;
  }
  
  public void test(NDArray[][] samples, int maxIterations, double convergence, int trials) {
    for (int epoch = 0; epoch < trials; epoch++)
    {
      PipelineNetwork net = this;
      double rms = 0;
      for (int i = 0; i < samples.length; i++) {
        NDArray input = samples[i][0];
        NDArray output = samples[i][1];
        rms += net.eval(input).errRms(output);
      }
      rms /= samples.length;
      TestNetworkUnit.log.info("RMS Error: {}", rms);
      for (int i = 0; i < maxIterations; i++)
      {
        if(shouldMutate(i,rms)){
          mutate();
        }
        rms = 0;
        for (int j = 0; j < samples.length; j++) {
          NDArray input = samples[j][0];
          NDArray output = samples[j][1];
          double rate = getRate(i);
          NNResult eval = net.eval(input);
          rms += eval.errRms(output);
          NDArray delta = eval.delta(rate, output);
          FeedbackContext ctx = new FeedbackContext();
          ctx.quantum = getQuantum();
          eval.feedback(delta, ctx);
        }
        rms /= samples.length;
        if (rms < convergence) break;
        if(isVerbose()) TestNetworkUnit.log.info("RMS Error: {}", rms);
      }
      TestNetworkUnit.log.info("RMS Error: {}", rms);
      if (rms >= convergence) {
        throw new RuntimeException("Failed in trial " + epoch);
      }
    }
  }
  private boolean verbose = false;

  protected boolean shouldMutate(int i, double rms) {
    //boolean r = (i%100)==0 && Math.random()<0.5;
    if((i%100)==0) {
      boolean r = (lastRms * .95) < rms;
      lastRms = rms;
      return r;
    }
    else
    {
      return false;
    }
  }

  protected void mutate() {
    layers.stream()
    .filter(l->(l instanceof DenseSynapseLayer))
    .forEach(l->mutate((DenseSynapseLayer)l));
  }

  protected DenseSynapseLayer mutate(DenseSynapseLayer l) {
    Random random = new Random();
    l.addWeights(() -> 0.005 * random.nextGaussian() * Math.exp(Math.random() * 4) / 2);
    //return l.freeze(new Random().nextBoolean());
    return l;
  }
  
  private double rate = 0.00001;
  
  public double getRate(int iteration) {
    double exp = Math.exp(Math.random()*Math.random()*4)/2;
    return rate * exp;
  }
  
  public double getRate() {
    return rate;
  }
  
  public PipelineNetwork setRate(double rate) {
    this.rate = rate;
    return this;
  }

  public double getQuantum() {
    return quantum;
  }

  public PipelineNetwork setQuantum(double quantum) {
    this.quantum = quantum;
    return this;
  }

  public boolean isVerbose() {
    return verbose;
  }

  public PipelineNetwork setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
}