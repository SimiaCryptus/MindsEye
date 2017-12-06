/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.network;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.NNExecutionContext;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.WeightExtractor;
import com.simiacryptus.mindseye.layers.java.WrapperLayer;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Directed Acyclical Graph Network
 * The base class for all conventional network wiring.
 */
public abstract class DAGNetwork extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);
  /**
   * The Input nodes.
   */
  public final LinkedHashMap<UUID, InputNode> inputNodes;
  /**
   * The Input handles.
   */
  public final List<UUID> inputHandles;
  /**
   * The Layers by id.
   */
  protected final LinkedHashMap<Object, NNLayer> layersById = new LinkedHashMap<>();
  /**
   * The Nodes by id.
   */
  protected final LinkedHashMap<UUID, DAGNode> nodesById = new LinkedHashMap<>();
  /**
   * The Labels.
   */
  protected final LinkedHashMap<String, UUID> labels = new LinkedHashMap<>();
  
  /**
   * Instantiates a new Dag network.
   *
   * @param json the json
   */
  protected DAGNetwork(JsonObject json) {
    super(json);
    inputHandles = new ArrayList<>();
    inputNodes = new LinkedHashMap<>();
    for (JsonElement item : json.getAsJsonArray("inputs")) {
      UUID key = UUID.fromString(item.getAsString());
      inputHandles.add(key);
      inputNodes.put(key, new InputNode(this, key));
    }
    JsonObject jsonNodes = json.getAsJsonObject("nodes");
    JsonObject jsonLayers = json.getAsJsonObject("layers");
    JsonObject jsonLinks = json.getAsJsonObject("links");
    JsonObject jsonLabels = json.getAsJsonObject("labels");
    Map<UUID, NNLayer> source_layersByNodeId = new HashMap<>();
    Map<UUID, NNLayer> source_layersByLayerId = new HashMap<>();
    for (Entry<String, JsonElement> e : jsonLayers.entrySet()) {
      source_layersByLayerId.put(UUID.fromString(e.getKey()), NNLayer.fromJson(e.getValue().getAsJsonObject()));
    }
    for (Entry<String, JsonElement> e : jsonNodes.entrySet()) {
      UUID nodeId = UUID.fromString(e.getKey());
      UUID layerId = UUID.fromString(e.getValue().getAsString());
      NNLayer layer = source_layersByLayerId.get(layerId);
      assert (null != layer);
      source_layersByNodeId.put(nodeId, layer);
    }
    final LinkedHashMap<String, UUID> labels = new LinkedHashMap<>();
    for (Entry<String, JsonElement> e : jsonLabels.entrySet()) {
      labels.put(e.getKey(), UUID.fromString(e.getValue().getAsString()));
    }
    Map<UUID, List<UUID>> deserializedLinks = new HashMap<>();
    for (Entry<String, JsonElement> e : jsonLinks.entrySet()) {
      ArrayList<UUID> linkList = new ArrayList<>();
      for (JsonElement linkItem : e.getValue().getAsJsonArray()) {
        linkList.add(UUID.fromString(linkItem.getAsString()));
      }
      deserializedLinks.put(UUID.fromString(e.getKey()), linkList);
    }
    for (UUID key : labels.values()) {
      initLinks(deserializedLinks, source_layersByNodeId, key);
    }
    UUID head = UUID.fromString(json.getAsJsonPrimitive("head").getAsString());
    initLinks(deserializedLinks, source_layersByNodeId, head);
    this.labels.putAll(labels);
    assertConsistent();
    for (NNLayer layer : source_layersByNodeId.values()) {
      if (layer instanceof WeightExtractor) {
        WeightExtractor weightExtractor = (WeightExtractor) layer;
        weightExtractor.setInner(source_layersByLayerId.get(weightExtractor.getInnerId()));
      }
    }
  }
  
  /**
   * Instantiates a new Dag network.
   *
   * @param inputs the inputs
   */
  public DAGNetwork(int inputs) {
    inputHandles = new ArrayList<>();
    inputNodes = new LinkedHashMap<>();
    for (int i = 0; i < inputs; i++) {
      addInput();
    }
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    JsonArray inputs = new JsonArray();
    json.add("inputs", inputs);
    inputHandles.forEach(uuid -> inputs.add(new JsonPrimitive(uuid.toString())));
    JsonObject layerMap = new JsonObject();
    JsonObject nodeMap = new JsonObject();
    JsonObject links = new JsonObject();
    nodesById.values().forEach(node -> {
      JsonArray linkArray = new JsonArray();
      Arrays.stream(node.getInputs()).forEach((DAGNode input) -> linkArray.add(new JsonPrimitive(input.getId().toString())));
      NNLayer layer = node.getLayer();
      String nodeId = node.getId().toString();
      String layerId = layer.getId().toString();
      nodeMap.addProperty(nodeId, layerId);
      layerMap.add(layerId, layer.getJson());
      links.add(nodeId, linkArray);
    });
    json.add("nodes", nodeMap);
    json.add("layers", layerMap);
    json.add("links", links);
    JsonObject labels = new JsonObject();
    this.labels.forEach((k, v) -> {
      labels.addProperty(k.toString(), v.toString());
    });
    json.add("labels", labels);
    json.addProperty("head", getHead().getId().toString());
    return json;
  }
  
  /**
   * Assert consistent boolean.
   *
   * @return the boolean
   */
  protected boolean assertConsistent() {
    assert null != getInput();
    for (Entry<String, UUID> e : labels.entrySet()) {
      assert (nodesById.containsKey(e.getValue()));
    }
    for (Entry<UUID, DAGNode> e : nodesById.entrySet()) {
      NNLayer layer = e.getValue().getLayer();
      assert (layersById.containsKey(layer.getId()));
      assert (layersById.get(layer.getId()) == layer);
    }
    return true;
  }
  
  private void initLinks(Map<UUID, List<UUID>> nodeLinks, Map<UUID, NNLayer> layersByNodeId, UUID newNodeId) {
    if (layersById.containsKey(newNodeId)) return;
    if (inputNodes.containsKey(newNodeId)) return;
    NNLayer layer = layersByNodeId.get(newNodeId);
    if (layer == null) {
      throw new IllegalArgumentException(String.format("%s is linked to but not defined", newNodeId));
    }
    List<UUID> links = nodeLinks.get(newNodeId);
    if (null != links) {
      for (UUID link : links) {
        initLinks(nodeLinks, layersByNodeId, link);
      }
    }
    assertConsistent();
    DAGNode[] dependencies = getDependencies(nodeLinks, newNodeId);
    final InnerNode node = new InnerNode(this, layer, newNodeId, dependencies);
    this.layersById.put(layer.getId(), layer);
    nodesById.put(node.getId(), node);
    assertConsistent();
  }
  
  private DAGNode[] getDependencies(Map<UUID, List<UUID>> deserializedLinks, UUID e) {
    List<UUID> links = deserializedLinks.get(e);
    if (null == links) return new DAGNode[]{};
    return links.stream().map(id -> getNode(id)).toArray(i -> new DAGNode[i]);
  }
  
  private DAGNode getNode(UUID id) {
    DAGNode returnValue = nodesById.get(id);
    if (null == returnValue) {
      returnValue = inputNodes.get(id);
    }
    return returnValue;
  }
  
  /**
   * Reset.
   */
  public void reset() {
    layersById.clear();
    nodesById.clear();
    labels.clear();
  }
  
  /**
   * Gets by name.
   *
   * @param <T>  the type parameter
   * @param name the name
   * @return the by name
   */
  public <T extends NNLayer> T getByName(String name) {
    if (null == name) return null;
    AtomicReference<NNLayer> result = new AtomicReference<>();
    visitLayers(n -> {
      if (name.equals(n.getName())) result.set(n);
    });
    return (T) result.get();
  }
  
  /**
   * Visit layers.
   *
   * @param visitor the visitor
   */
  public void visitLayers(Consumer<NNLayer> visitor) {
    layersById.values().forEach(layer -> {
      if (layer instanceof DAGNetwork) {
        ((DAGNetwork) layer).visitLayers(visitor);
      }
      if (layer instanceof WrapperLayer) {
        visitor.accept(((WrapperLayer) layer).getInner());
      }
      visitor.accept(layer);
    });
  }
  
  /**
   * Visit nodes.
   *
   * @param visitor the visitor
   */
  public void visitNodes(Consumer<DAGNode> visitor) {
    nodesById.values().forEach(node -> {
      if (node.getLayer() instanceof DAGNetwork) {
        ((DAGNetwork) node.getLayer()).visitNodes(visitor);
      }
      visitor.accept(node);
    });
  }
  
  /**
   * Attach.
   *
   * @param obj the obj
   */
  public void attach(MonitoredObject obj) {
    visitLayers(layer -> {
      if (layer instanceof MonitoredItem) {
        obj.addObj(layer.getName(), (MonitoredItem) layer);
      }
    });
  }
  
  /**
   * Add input nn layer.
   *
   * @return the nn layer
   */
  public NNLayer addInput() {
    UUID key = UUID.randomUUID();
    inputHandles.add(key);
    inputNodes.put(key, new InputNode(this, key));
    return this;
  }
  
  /**
   * Remove last input nn layer.
   *
   * @return the nn layer
   */
  public NNLayer removeLastInput() {
    int index = inputHandles.size() - 1;
    UUID key = inputHandles.remove(index);
    inputNodes.remove(key);
    return this;
  }
  
  /**
   * Gets nodes.
   *
   * @return the nodes
   */
  public List<DAGNode> getNodes() {
    return Stream.concat(
      nodesById.values().stream(),
      getInput().stream()
    ).collect(Collectors.toList());
  }
  
  /**
   * Single exe ctx graph evaluation context.
   *
   * @param input the input
   * @return the graph evaluation context
   */
  public final GraphEvaluationContext singleExeCtx(final Tensor... input) {
    return buildExeCtx(NNResult.singleResultArray(input));
  }
  
  /**
   * Build exe ctx graph evaluation context.
   *
   * @param inputs the inputs
   * @return the graph evaluation context
   */
  public GraphEvaluationContext buildExeCtx(final NNResult... inputs) {
    assert (inputs.length == inputHandles.size()) : inputs.length + " != " + inputHandles.size();
    final GraphEvaluationContext graphEvaluationContext = new GraphEvaluationContext();
    for (int i = 0; i < inputs.length; i++) {
      graphEvaluationContext.cache.put(this.inputHandles.get(i), new CountingNNResult(inputs[i]));
    }
    return graphEvaluationContext;
  }
  
  /**
   * Batch exe context graph evaluation context.
   *
   * @param batchData the batch data
   * @return the graph evaluation context
   */
  public GraphEvaluationContext batchExeContext(final Tensor[][] batchData) {
    return this.buildExeCtx(NNResult.batchResultArray(batchData));
  }
  
  /**
   * Gets child node.
   *
   * @param id the id
   * @return the child node
   */
  public DAGNode getChildNode(final UUID id) {
    if (this.nodesById.containsKey(id)) {
      return this.nodesById.get(id);
    }
    return this.nodesById.values().stream().map(x -> x.getLayer())
      .filter(x -> x instanceof DAGNetwork)
      .map(x -> ((DAGNetwork) x).getChildNode(id)).findAny().orElse(null);
  }
  
  @Override
  public List<NNLayer> getChildren() {
    return this.layersById.values().stream().flatMap(l -> l.getChildren().stream()).distinct().sorted(Comparator.comparing(l -> l.getId().toString())).collect(Collectors.toList());
  }
  
  /**
   * Gets input.
   *
   * @return the input
   */
  public List<DAGNode> getInput() {
    ArrayList<DAGNode> list = new ArrayList<>();
    for (UUID key : inputHandles) list.add(inputNodes.get(key));
    return list;
  }
  
  @Override
  public List<double[]> state() {
    return getChildren().stream().flatMap(l -> l.state().stream()).distinct().collect(Collectors.toList());
  }
  
  /**
   * Gets head.
   *
   * @return the head
   */
  public abstract DAGNode getHead();
  
  /**
   * Gets by label.
   *
   * @param key the key
   * @return the by label
   */
  public DAGNode getByLabel(String key) {
    return nodesById.get(labels.get(key));
  }
  
  /**
   * Gets label network.
   *
   * @param key the key
   * @return the label network
   */
  public NNLayer getLabelNetwork(String key) {
    return new NNLayer() {
      @Override
      public NNResult eval(NNExecutionContext nncontext, NNResult[] array) {
        return nodesById.get(labels.get(key)).get(nncontext, buildExeCtx(array));
      }
      
      @Override
      public JsonObject getJson() {
        throw new UnsupportedOperationException();
      }
      
      @Override
      public List<double[]> state() {
        return DAGNetwork.this.state();
      }
    };
  }
  
  /**
   * Get nn result.
   *
   * @param nncontext   the nncontext
   * @param buildExeCtx the build exe ctx
   * @return the nn result
   */
  public NNResult get(NNExecutionContext nncontext, GraphEvaluationContext buildExeCtx) {
    return getHead().get(nncontext, buildExeCtx);
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult[] input) {
    return getHead().get(nncontext, buildExeCtx(input));
  }
  
  @Override
  public DAGNetwork copy() {
    return (DAGNetwork) super.copy();
  }
  
  @Override
  public DAGNetwork setFrozen(boolean frozen) {
    super.setFrozen(frozen);
    if (null != layersById) visitLayers(layer -> layer.setFrozen(frozen));
    return this;
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @param head     the head
   * @return the dag node
   */
  public DAGNode add(final NNLayer nextHead, final DAGNode... head) {
    return add(null, nextHead, head);
  }
  
  /**
   * Add dag node.
   *
   * @param label the label
   * @param layer the layer
   * @param head  the head
   * @return the dag node
   */
  public DAGNode add(String label, final NNLayer layer, final DAGNode... head) {
    assertConsistent();
    assert null != getInput();
    final InnerNode node = new InnerNode(this, layer, head);
    this.layersById.put(layer.getId(), layer);
    nodesById.put(node.getId(), node);
    if (null != label) labels.put(label, node.getId());
    assertConsistent();
    return node;
  }
  
  /**
   * Gets input.
   *
   * @param index the index
   * @return the input
   */
  public DAGNode getInput(int index) {
    DAGNode input = inputNodes.get(inputHandles.get(index));
    assert null != input;
    return input;
  }
  
  /**
   * Gets layer.
   *
   * @return the layer
   */
  public NNLayer getLayer() {
    return this;
  }
  
}
