# StdDevMetaLayer
## StdDevMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f9e",
      "isFrozen": false,
      "name": "StdDevMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f9e",
      "inputs": [
        "88146096-8003-4f2b-a5b8-f3e83c8559c2"
      ],
      "nodes": {
        "87709b2a-7c49-4859-aa0c-32b5df380dca": "c88cbdf1-1c2a-4a5e-b964-890900000fa2",
        "0d123abf-d1a9-4f22-a14b-92a54aa87b4b": "c88cbdf1-1c2a-4a5e-b964-890900000fa1",
        "68015413-fbe2-4a51-b92f-efe94ced9d69": "c88cbdf1-1c2a-4a5e-b964-890900000fa5",
        "24a3776e-2129-4d44-9c19-a14d44e3ce31": "c88cbdf1-1c2a-4a5e-b964-890900000fa4",
        "c5d47345-ab50-4d8b-8693-81b63dadc054": "c88cbdf1-1c2a-4a5e-b964-890900000fa3",
        "1187b106-6bd7-4443-9cba-f787da4f1632": "c88cbdf1-1c2a-4a5e-b964-890900000fa0",
        "c39436e3-da5e-4cbd-9b58-071adfe3f2e1": "c88cbdf1-1c2a-4a5e-b964-890900000f9f"
      },
      "layers": {
        "c88cbdf1-1c2a-4a5e-b964-890900000fa2": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000fa2",
          "isFrozen": true,
          "name": "SqActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000fa2"
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000fa1": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000fa1",
          "isFrozen": false,
          "name": "AvgMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000fa1"
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000fa5": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000fa5",
          "isFrozen": false,
          "name": "AvgMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000fa5"
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000fa4": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000fa4",
          "isFrozen": true,
          "name": "SqActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000fa4"
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000fa3": {
          "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000fa3",
          "isFrozen": false,
          "name": "LinearActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000fa3",
          "weights": {
            "dimensions": [
              2
            ],
            "data": [
              -1.0,
              0.0
            ]
          }
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000fa0": {
          "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000fa0",
          "isFrozen": false,
          "name": "SumInputsLayer/c88cbdf1-1c2a-4a5e-b964-890900000fa0"
        },
        "c88cbdf1-1c2a-4a5e-b964-890900000f9f": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-890900000f9f",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/c88cbdf1-1c2a-4a5e-b964-890900000f9f",
          "power": 0.5
        }
      },
      "links": {
        "87709b2a-7c49-4859-aa0c-32b5df380dca": [
          "88146096-8003-4f2b-a5b8-f3e83c8559c2"
        ],
        "0d123abf-d1a9-4f22-a14b-92a54aa87b4b": [
          "87709b2a-7c49-4859-aa0c-32b5df380dca"
        ],
        "68015413-fbe2-4a51-b92f-efe94ced9d69": [
          "88146096-8003-4f2b-a5b8-f3e83c8559c2"
        ],
        "24a3776e-2129-4d44-9c19-a14d44e3ce31": [
          "68015413-fbe2-4a51-b92f-efe94ced9d69"
        ],
        "c5d47345-ab50-4d8b-8693-81b63dadc054": [
          "24a3776e-2129-4d44-9c19-a14d44e3ce31"
        ],
        "1187b106-6bd7-4443-9cba-f787da4f1632": [
          "0d123abf-d1a9-4f22-a14b-92a54aa87b4b",
          "c5d47345-ab50-4d8b-8693-81b63dadc054"
        ],
        "c39436e3-da5e-4cbd-9b58-071adfe3f2e1": [
          "1187b106-6bd7-4443-9cba-f787da4f1632"
        ]
      },
      "labels": {},
      "head": "c39436e3-da5e-4cbd-9b58-071adfe3f2e1"
    }
```



### Network Diagram
Code from [LayerTestBase.java:95](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L95) executed in 0.21 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ -0.192, 1.96, -1.748 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.0 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: StdDevMetaLayer/c88cbdf1-1c2a-4a5e-b964-890900000f9e
    Inputs: [ -0.192, 1.96, -1.748 ]
    output=[ 0.0, 0.0, 0.0 ]
    measured/actual: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    implemented/expected: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.3607 +- 0.0676 [0.2992 - 0.6782]
    Learning performance: 0.1018 +- 0.0509 [0.0684 - 0.4987]
    
```

