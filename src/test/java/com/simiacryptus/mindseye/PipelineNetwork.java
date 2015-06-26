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
        TestNetworkUnit.log.info("RMS Error: {}", rms);
        if (rms < convergence) break;
      }
      if (rms >= convergence) {
        throw new RuntimeException("Failed in trial " + epoch);
      }
    }
  }

  protected boolean shouldMutate(int i, double rms) {
    //boolean r = (i%100)==0 && Math.random()<0.5;
    boolean r = (i%10)==0 && (lastRms * .9) < rms;
    if(r) {
      lastRms = rms;
      return true;
    }
    else
    {
      lastRms = rms;
      return false;
    }
  }

  protected void mutate() {
    layers.stream()
    .filter(l->(l instanceof DenseSynapseLayer))
    .map(l->(DenseSynapseLayer)l)
    .forEach(l->mutate(l));
  }

  protected DenseSynapseLayer mutate(DenseSynapseLayer l) {
    return l.freeze(new Random().nextBoolean());
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
  
}