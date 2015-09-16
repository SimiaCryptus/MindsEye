package com.simiacryptus.mindseye.net.dag;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.training.Tester;

import groovy.lang.Tuple2;

/***
 * Builds a network NNLayer components, assumed to form a directed acyclic graph
 * with a single output. Supplied builder methods designed to build linear
 * sequence of units acting on the current output node.
 *
 * @author Andrew Charneski
 */
public class DAGNetwork extends NNLayer<DAGNetwork> {
  private final class InputNode extends LazyResult {
    @Override
    protected NNResult[] eval(final EvaluationContext t) {
      return t.cache.get(DAGNetwork.this.inputHandle);
    }

    @Override
    protected JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.addProperty("target", DAGNetwork.this.inputHandle.toString());
      return json;
    }
  }

  private final class UnaryNode extends LazyResult {
    private final UUID layer;
    private final LazyResult prevHead;

    private UnaryNode(final NNLayer<?> layer, final LazyResult prevHead) {
      this.layer = layer.getId();
      this.prevHead = prevHead;
    }

    @Override
    protected NNResult[] eval(final EvaluationContext ctx) {
      final NNResult[] input = this.prevHead.get(ctx);
      final NNResult output = DAGNetwork.this.byId.get(this.layer).eval(ctx, input);
      return new NNResult[] { output };
    }

    @Override
    protected JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.add("layer", DAGNetwork.this.byId.get(this.layer).getJson());
      json.add("prev", this.prevHead.toJson());
      return json;
    }
  }

  private final class BinaryNode extends LazyResult {
    private final UUID layer;
    private final LazyResult left;
    private final LazyResult right;

    private BinaryNode(final NNLayer<?> layer, final LazyResult left, final LazyResult right) {
      this.layer = layer.getId();
      this.left = left;
      this.right = right;
    }

    @Override
    protected NNResult[] eval(final EvaluationContext ctx) {
      final NNResult inputL = this.left.get(ctx)[0];
      final NNResult inputR = this.right.get(ctx)[1];
      final NNResult output = DAGNetwork.this.byId.get(this.layer).eval(ctx, inputL, inputR);
      return new NNResult[] { output };
    }

    @Override
    protected JsonObject toJson() {
      final JsonObject json = new JsonObject();
      json.add("layer", DAGNetwork.this.byId.get(this.layer).getJson());
      json.add("left", this.left.toJson());
      json.add("right", this.right.toJson());
      return json;
    }
  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);

  private final java.util.LinkedHashMap<UUID, NNLayer<?>> byId = new java.util.LinkedHashMap<>();
  public final InputNode inputNode = new InputNode();
  private LazyResult head = inputNode;
  public final UUID inputHandle = UUID.randomUUID();

  private final java.util.HashMap<NNLayer<?>, NNLayer<?>> nextMap = new java.util.HashMap<>();
  private final java.util.HashMap<NNLayer<?>, NNLayer<?>> prevMap = new java.util.HashMap<>();

  public synchronized DAGNetwork add(final NNLayer<?> nextHead) {
    this.byId.put(nextHead.getId(), nextHead);
    final NNLayer<?> prevHead = getHeadLayer();
    this.prevMap.put(nextHead, prevHead);
    this.nextMap.put(prevHead, nextHead);
    UnaryNode node = new UnaryNode(nextHead, getHead());
    setHead(node);
    return this;
  }

  public synchronized DAGNetwork add2(final NNLayer<?> nextHead) {
    return add2(nextHead, inputNode);
  }

  public synchronized DAGNetwork add2(final NNLayer<?> nextHead, InputNode right) {
    this.byId.put(nextHead.getId(), nextHead);
    final NNLayer<?> prevHead = getHeadLayer();
    this.prevMap.put(nextHead, prevHead);
    this.nextMap.put(prevHead, nextHead);
    setHead(new BinaryNode(nextHead, getHead(), right));
    return this;
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... array) {
    evaluationContext.cache.put(this.inputHandle, array);
    return getHead().get(evaluationContext)[0];
  }

  public NNResult eval(final NDArray... array) {
    return eval(new EvaluationContext(), array);
  }

  @Override
  public NNLayer<?> evolve() {
    if (0 == this.byId.values().stream().filter(l -> {
      final NNLayer<?> evolve = l.evolve();
      if (null != evolve && evolve != l)
        throw new RuntimeException("Not implemented: Substitution via evolution in DAGNetwork");
      return null != evolve;
    }).count())
      return null;
    else
      return this;
  }

  @Override
  public DAGNetwork freeze() {
    this.byId.values().forEach(l -> l.freeze());
    return super.freeze();
  }

  public NNLayer<?> get(final int i) {
    return this.byId.get(i);
  }

  @Override
  public List<NNLayer<?>> getChildren() {
    return this.byId.values().stream().flatMap(l -> l.getChildren().stream()).distinct().sorted(Comparator.comparing(l -> l.getId())).collect(Collectors.toList());
  }

  public LazyResult getHead() {
    return this.head;
  }

  public NNLayer<?> getHeadLayer() {
    if (this.head instanceof UnaryNode)
      return DAGNetwork.this.byId.get(((UnaryNode) this.head).layer);
    else if (this.head instanceof BinaryNode)
      return DAGNetwork.this.byId.get(((BinaryNode) this.head).layer);
    else return null;
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.add("root", getHead().toJson());
    // for(NNLayer c : getChildren()){
    // json.add(c.getId(), c.getJson());
    // }
    return json;
  }

  private void permutate_back(final NNLayer<?> permutationLayer, final List<Tuple2<Integer, Integer>> permute) {
    final NNLayer<?> prev = this.prevMap.get(permutationLayer);
    final List<Tuple2<Integer, Integer>> passback = prev.permuteOutput(permute);
    if (null != passback) {
      permutate_back(prev, passback);
    }
  }

  private void permutate_forward(final NNLayer<?> permutationLayer, final List<Tuple2<Integer, Integer>> permute) {
    final NNLayer<?> next = this.nextMap.get(permutationLayer);
    final List<Tuple2<Integer, Integer>> passforward = next.permuteInput(permute);
    if (null != passforward) {
      permutate_forward(next, passforward);
    }
  }

  public void permute(final UUID id, final List<Tuple2<Integer, Integer>> permute) {
    final NNLayer<?> permutationLayer = this.byId.get(id);
    permutate_back(permutationLayer, permute);
    permutate_forward(permutationLayer, permute);

  }

  public void setHead(final LazyResult head) {
    this.head = head;
  }

  @Override
  public List<double[]> state() {
    return getChildren().stream().flatMap(l -> l.state().stream()).distinct().collect(Collectors.toList());
  }

  public Tester trainer(final NDArray[][] samples) {
    return new Tester().setParams(this, samples);
  }

  @Override
  public NNLayer<?> getChild(UUID id) {
    if(this.id.equals(id)) return this;
    if(byId.containsKey(id)) return byId.get(id);
    return this.byId.values().stream().map(x->x.getChild(id)).findAny().orElse(null);
  }

}
