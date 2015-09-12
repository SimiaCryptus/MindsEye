package com.simiacryptus.mindseye.net.dev;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;

public class TreeNodeFunctionalLayer extends NNLayer {

  private static final class NNResultBuffer extends NNResult {
    private NNResult inner;

    private NNResultBuffer(NNResult x) {
      super(x.data);
      this.inner = x;
    }

    @Override
    public boolean isAlive() {
      return inner.isAlive();
    }

    LogNDArray sum = null;
    @Override
    public synchronized void feedback(LogNDArray data, DeltaBuffer buffer) {
      if (null == sum) {
        sum = new LogNDArray(data.getDims());
      }
      sum = sum.add(data);
    }

    public void flush(DeltaBuffer buffer) {
      inner.feedback(sum, buffer);
    }
  }

  private static final Logger log = LoggerFactory.getLogger(TreeNodeFunctionalLayer.class);

  private boolean frozen = false;
  private boolean verbose = false;

  private NNLayer gate;
  private List<NNLayer> leafs;

  protected TreeNodeFunctionalLayer(final NNLayer gate, List<NNLayer> leafs) {
    super();
    this.gate = gate;
    this.leafs = leafs;
  }

  public TreeNodeFunctionalLayer(final NNLayer gate, NNLayer... leafs) {
    this(gate,java.util.stream.Stream.of(leafs).collect(java.util.stream.Collectors.toList()));
  }

  public TreeNodeFunctionalLayer(final NNLayer gate, int count, java.util.function.IntFunction<NNLayer> leafs) {
    this(gate, IntStream.range(0, count).mapToObj(x->leafs.apply(x)).collect(java.util.stream.Collectors.toList()));
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj2) {
    List<NNResultBuffer> inputResultBuffers = java.util.stream.Stream.of(inObj2)
        .map(x->new NNResultBuffer(x)).collect(java.util.stream.Collectors.toList());
    NNResult[] inObj = inputResultBuffers.stream().toArray(i->new NNResult[i]);
    NNResult gateEval = this.gate.eval(evaluationContext, inObj);
    double[] gateVals = gateEval.data.getData();
//    int[] sorted = IntStream.range(0, gateVals.length).mapToObj(x->x)
//        .sorted(java.util.Comparator.comparing(i->gateVals[i])).mapToInt(x->x).toArray();
    
    List<NNResult> outputs = IntStream.range(0, gateVals.length).mapToObj(x->{
      NNLayer leaf = leafs.get(x);
      NNResult eval = leaf.eval(evaluationContext, inObj);
      return scale(eval, gateVals[x]);
    }).collect(java.util.stream.Collectors.toList());
    NNResult output = outputs.stream().reduce((l,r)->add(l,r)).get();
    if (isVerbose()) {
      TreeNodeFunctionalLayer.log.debug(String.format("Feed forward: %s * %s => %s", inObj[0].data, gateEval.data, output));
    }
    return new NNResult(output.data) {
      
      @Override
      public boolean isAlive() {
        return output.isAlive() || java.util.stream.Stream.of(inObj).anyMatch(x -> x.isAlive());
      }
      
      @Override
      public void feedback(LogNDArray data, DeltaBuffer buffer) {
        output.feedback(data, buffer);
        NDArray evalFeedback = new NDArray(gateEval.data.getDims());
        for(int subnet=0;subnet<outputs.size();subnet++) {
          NNResult subnetObj = outputs.get(subnet);
          NDArray so = subnetObj.data;
          double sum1 = 0;
          for(int i=0;i<so.dim();i++){
            sum1 += data.getData()[i].multiply(so.getData()[i]).doubleValue();
          }
          evalFeedback.set(subnet, sum1);
        }
        gateEval.feedback(evalFeedback.log(), buffer);
        inputResultBuffers.stream().forEach(x->x.flush(buffer));
      }
    };
  }

  private NNResult add(NNResult a, NNResult b) {
    return new NNResult(a.data.add(b.data)) {
      
      @Override
      public boolean isAlive() {
        return a.isAlive()||b.isAlive();
      }
      
      @Override
      public void feedback(LogNDArray data, DeltaBuffer buffer) {
        a.feedback(data, buffer);
        b.feedback(data, buffer);
      }
    };
  }

  private NNResult scale(NNResult eval, double d) {
    return new NNResult(eval.data.scale(d)) {
      
      @Override
      public boolean isAlive() {
        return eval.isAlive();
      }
      
      @Override
      public void feedback(LogNDArray data, DeltaBuffer buffer) {
        eval.feedback(data.scale(d), buffer);
      }
    };
  }

  public TreeNodeFunctionalLayer freeze() {
    return freeze(true);
  }

  public TreeNodeFunctionalLayer freeze(final boolean b) {
    this.frozen = b;
    return this;
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.add("gate", this.gate.getJson());
    com.google.gson.JsonArray childArray = new com.google.gson.JsonArray();
    this.leafs.stream().forEach(x->childArray.add(x.getJson()));
    json.add("children", childArray);
    return json;
  }

  public boolean isFrozen() {
    return this.frozen;
  }

  @Override
  public boolean isVerbose() {
    return this.verbose;
  }

  public TreeNodeFunctionalLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

  public TreeNodeFunctionalLayer thaw() {
    return freeze(false);
  }

  public List<NNLayer> getChildren() {
    ArrayList<NNLayer> r = new java.util.ArrayList<>();
    r.addAll(gate.getChildren());
    this.leafs.stream().forEach(x->r.addAll(x.getChildren()));
    return r;
  }
}
