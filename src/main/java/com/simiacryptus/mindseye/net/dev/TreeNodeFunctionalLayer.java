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
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;

public class TreeNodeFunctionalLayer extends NNLayer<TreeNodeFunctionalLayer> {

  private static final class NNResultBuffer extends NNResult {
    private NNResult inner;

    NDArray sum = null;

    private NNResultBuffer(final NNResult x) {
      super(x.evaluationContext, x.data);
      this.inner = x;
    }

    @Override
    public synchronized void feedback(final NDArray data, final DeltaBuffer buffer) {
      if (null == data)
        return;
      if (null == this.sum) {
        this.sum = new NDArray(data.getDims());
      }
      this.sum = this.sum.add(data);
    }

    public void flush(final DeltaBuffer buffer) {
      this.inner.feedback(this.sum, buffer);
    }

    @Override
    public boolean isAlive() {
      return this.inner.isAlive();
    }
  }

  private static final Logger log = LoggerFactory.getLogger(TreeNodeFunctionalLayer.class);

  private NNLayer<?> gate;
  private List<NNLayer<?>> leafs;

  public TreeNodeFunctionalLayer(final NNLayer<?> gate, final int count, final java.util.function.IntFunction<NNLayer<?>> leafs) {
    this(gate, IntStream.range(0, count).mapToObj(x -> leafs.apply(x)).collect(java.util.stream.Collectors.toList()));
  }

  protected TreeNodeFunctionalLayer(final NNLayer<?> gate, final List<NNLayer<?>> leafs) {
    super();
    this.gate = gate;
    this.leafs = leafs;
  }

  public TreeNodeFunctionalLayer(final NNLayer<?> gate, final NNLayer<?>... leafs) {
    this(gate, java.util.stream.Stream.of(leafs).collect(java.util.stream.Collectors.toList()));
  }

  private NNResult add(final NNResult a, final NNResult b) {
    assert(a.evaluationContext==b.evaluationContext);
    return new NNResult(a.evaluationContext, a.data.add(b.data)) {

      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        a.feedback(data, buffer);
        b.feedback(data, buffer);
      }

      @Override
      public boolean isAlive() {
        return a.isAlive() || b.isAlive();
      }
    };
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj2) {
    final List<NNResultBuffer> inputResultBuffers = java.util.stream.Stream.of(inObj2)
        .map(x -> new NNResultBuffer(x)).collect(java.util.stream.Collectors.toList());
    final NNResult[] inObj = inputResultBuffers.stream().toArray(i -> new NNResult[i]);
    final NNResult gateEval = this.gate.eval(evaluationContext, inObj);
    final double[] gateVals = gateEval.data.getData();
    // int[] sorted = IntStream.range(0, gateVals.length).mapToObj(x->x)
    // .sorted(java.util.Comparator.comparing(i->gateVals[i])).mapToInt(x->x).toArray();

    final List<NNResult> outputs = IntStream.range(0, gateVals.length).mapToObj(x -> {
      final NNLayer<?> leaf = this.leafs.get(x);
      final NNResult eval = leaf.eval(evaluationContext, inObj);
      return scale(eval, gateVals[x]);
    }).collect(java.util.stream.Collectors.toList());
    final NNResult output = outputs.stream().reduce((l, r) -> add(l, r)).get();
    if (isVerbose()) {
      TreeNodeFunctionalLayer.log.debug(String.format("Feed forward: %s * %s => %s", inObj[0].data, gateEval.data, output));
    }
    return new NNResult(evaluationContext, output.data) {

      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        output.feedback(data, buffer);
        final NDArray evalFeedback = new NDArray(gateEval.data.getDims());
        for (int subnet = 0; subnet < outputs.size(); subnet++) {
          final NNResult subnetObj = outputs.get(subnet);
          final NDArray so = subnetObj.data;
          double sum1 = 0;
          for (int i = 0; i < so.dim(); i++) {
            sum1 += data.getData()[i] * so.getData()[i];
          }
          evalFeedback.set(subnet, sum1);
        }
        gateEval.feedback(evalFeedback, buffer);
        inputResultBuffers.stream().forEach(x -> x.flush(buffer));
      }

      @Override
      public boolean isAlive() {
        return output.isAlive() || java.util.stream.Stream.of(inObj).anyMatch(x -> x.isAlive());
      }
    };
  }

  @Override
  public List<NNLayer<?>> getChildren() {
    final ArrayList<NNLayer<?>> r = new java.util.ArrayList<>();
    r.addAll(this.gate.getChildren());
    this.leafs.stream().forEach(x -> r.addAll(x.getChildren()));
    return r;
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.add("gate", this.gate.getJson());
    final com.google.gson.JsonArray childArray = new com.google.gson.JsonArray();
    this.leafs.stream().forEach(x -> childArray.add(x.getJson()));
    json.add("children", childArray);
    return json;
  }

  private static NNResult scale(final NNResult eval, final double d) {
    return new NNResult(eval.evaluationContext, eval.data.scale(d)) {

      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        eval.feedback(data.scale(d), buffer);
      }

      @Override
      public boolean isAlive() {
        return eval.isAlive();
      }
    };
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
