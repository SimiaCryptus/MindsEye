# StdDevMetaLayer
## StdDevMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.StdDevMetaLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00003672",
      "isFrozen": false,
      "name": "StdDevMetaLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00003672",
      "inputs": [
        "a3272767-6301-447a-8ecc-1449b98f228b"
      ],
      "nodes": {
        "0b7599e9-3e09-45b6-94bb-b907e5394ada": "e2a3bda5-e7e7-4c05-aeb3-4ede00003676",
        "fb751a5e-c7de-46cc-b51e-2ed92ad9a8b3": "e2a3bda5-e7e7-4c05-aeb3-4ede00003675",
        "c917e314-abbc-4a0d-a07a-3bacb8901e3e": "e2a3bda5-e7e7-4c05-aeb3-4ede00003679",
        "fc8c2f9e-ff03-4cb4-a1e2-927f53004166": "e2a3bda5-e7e7-4c05-aeb3-4ede00003678",
        "faab0e16-4012-41d8-a9ee-7a858b37d6fb": "e2a3bda5-e7e7-4c05-aeb3-4ede00003677",
        "e4e74e8b-a37b-40b0-a69a-486a291d2496": "e2a3bda5-e7e7-4c05-aeb3-4ede00003674",
        "b1db412e-7d74-4db9-9e3a-c4ddca1d851e": "e2a3bda5-e7e7-4c05-aeb3-4ede00003673"
      },
      "layers": {
        "e2a3bda5-e7e7-4c05-aeb3-4ede00003676": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "e2a3b
```
...[skipping 1699 bytes](etc/91.txt)...
```
    hPowerActivationLayer",
          "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00003673",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00003673",
          "power": 0.5
        }
      },
      "links": {
        "0b7599e9-3e09-45b6-94bb-b907e5394ada": [
          "a3272767-6301-447a-8ecc-1449b98f228b"
        ],
        "fb751a5e-c7de-46cc-b51e-2ed92ad9a8b3": [
          "0b7599e9-3e09-45b6-94bb-b907e5394ada"
        ],
        "c917e314-abbc-4a0d-a07a-3bacb8901e3e": [
          "a3272767-6301-447a-8ecc-1449b98f228b"
        ],
        "fc8c2f9e-ff03-4cb4-a1e2-927f53004166": [
          "c917e314-abbc-4a0d-a07a-3bacb8901e3e"
        ],
        "faab0e16-4012-41d8-a9ee-7a858b37d6fb": [
          "fc8c2f9e-ff03-4cb4-a1e2-927f53004166"
        ],
        "e4e74e8b-a37b-40b0-a69a-486a291d2496": [
          "fb751a5e-c7de-46cc-b51e-2ed92ad9a8b3",
          "faab0e16-4012-41d8-a9ee-7a858b37d6fb"
        ],
        "b1db412e-7d74-4db9-9e3a-c4ddca1d851e": [
          "e4e74e8b-a37b-40b0-a69a-486a291d2496"
        ]
      },
      "labels": {},
      "head": "b1db412e-7d74-4db9-9e3a-c4ddca1d851e"
    }
```



### Network Diagram
Code from [LayerTestBase.java:132](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.18 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.53.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ 1.688, -0.568, -0.204 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.0 ]
```



### Performance
Code from [LayerTestBase.java:192](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L192) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.4934 +- 0.1783 [0.3220 - 1.5959]
    Learning performance: 0.0087 +- 0.0043 [0.0057 - 0.0371]
    
```

