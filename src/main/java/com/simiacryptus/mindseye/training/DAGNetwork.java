package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext.LazyResult;

/***
 * Builds a network NNLayer components, assumed to form a directed acyclic graph with a single output.
 * Supplied builder methods designed to build linear sequence of units acting on the current output node.
 *
 * @author Andrew Charneski
 */
public class DAGNetwork extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);

  private final List<NNLayer> children = new ArrayList<NNLayer>();

  public LazyResult<NNResult[]> head = new LazyResult<NNResult[]>() {
    @Override
    protected NNResult[] initialValue(final EvaluationContext t) {
      return (NNResult[]) t.cache.get(DAGNetwork.this.inputHandle);
    }

    @Override
    protected JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.addProperty("target", DAGNetwork.this.inputHandle.toString());
      return json;
    }
  };
  public final UUID inputHandle = UUID.randomUUID();

  public synchronized DAGNetwork add(final NNLayer layer) {
    this.children.add(layer);
    final LazyResult<NNResult[]> prevHead = this.head;
    this.head = new LazyResult<NNResult[]>() {
      @Override
      protected NNResult[] initialValue(final EvaluationContext ctx) {
        final NNResult[] input = prevHead.get(ctx);
        final NNResult output = layer.eval(ctx, input);
        return new NNResult[] { output };
      }

      @Override
      protected JsonObject toJson() {
        final JsonObject json = new JsonObject();
        json.add("layer", layer.getJson());
        json.add("prev", prevHead.toJson());
        return json;
      }
    };
    return this;
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... array) {
    evaluationContext.cache.put(this.inputHandle, array);
    return this.head.get(evaluationContext)[0];
  }

  public NNResult eval(final NDArray... array) {
    return eval(new EvaluationContext(), array);
  }

  public NNLayer get(final int i) {
    return this.children.get(i);
  }

  @Override
  public List<NNLayer> getChildren() {
    return this.children.stream().flatMap(l -> l.getChildren().stream()).distinct().sorted(Comparator.comparing(l -> l.getId())).collect(Collectors.toList());
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.add("root", this.head.toJson());
    // for(NNLayer c : getChildren()){
    // json.add(c.getId(), c.getJson());
    // }
    return json;
  }

  public Tester trainer(final NDArray[][] samples) {
    return new Tester().setParams(this, samples);
  }

}
