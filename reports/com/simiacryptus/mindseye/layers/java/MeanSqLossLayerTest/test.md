# MeanSqLossLayer
## MeanSqLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "f4569375-56fe-4e46-925c-95f400000a29",
      "isFrozen": false,
      "name": "MeanSqLossLayer/f4569375-56fe-4e46-925c-95f400000a29"
    }
```



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
    [[ 0.3, -0.82, 0.948, 0.896 ],
    [ -1.804, 1.672, -1.188, 0.356 ]]
    --------------------
    Output: 
    [ 3.8727440000000004 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 0.3, -0.82, 0.948, 0.896 ],
    [ -1.804, 1.672, -1.188, 0.356 ]
    Output: [ 3.8727440000000004 ]
    Measured: [ [ 1.0520249999945719 ], [ -1.2459750000015646 ], [ 1.0680249999994729 ], [ 0.27002499999984053 ] ]
    Implemented: [ [ 1.052 ], [ -1.246 ], [ 1.068 ], [ 0.27 ] ]
    Error: [ [ 2.4999994571839323E-5 ], [ 2.499999843541545E-5 ], [ 2.4999999472807843E-5 ], [ 2.499999984051371E-5 ] ]
    Feedback for input 1
    Inputs: [ 0.3, -0.82, 0.948, 0.896 ],
    [ -1.804, 1.672, -1.188, 0.356 ]
    Output: [ 3.8727440000000004 ]
    Measured: [ [ -1.0519749999993167 ], [ 1.2460249999968198 ], [ -1.0679749999997767 ], [ -0.2699750000001444 ] ]
    Implemented: [ [ -1.052 ], [ 1.246 ], [ -1.068 ], [ -0.27 ] ]
    Error: [ [ 2.500000068339503E-5 ], [ 2.4999996819818904E-5 ], [ 2.5000000223318608E-5 ], [ 2.4999999855612742E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5000e-05 +- 1.9293e-12 [2.5000e-05 - 2.5000e-05] (8#)
    relativeTol: 1.9979e-05 +- 1.5212e-05 [1.0032e-05 - 4.6298e-05] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2022 +- 0.1000 [0.1225 - 0.7267]
    Learning performance: 0.0036 +- 0.0024 [0.0000 - 0.0171]
    
```

