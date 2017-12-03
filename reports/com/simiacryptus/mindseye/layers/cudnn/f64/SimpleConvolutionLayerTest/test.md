# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000143a",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/e2d0bffa-47dc-4875-864f-3d3d0000143a",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          0.14
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    [[
    	[ [ -0.944 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.13216 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000143b",
      "isFrozen": false,
      "name": "ConvolutionLayer/e2d0bffa-47dc-4875-864f-3d3d0000143b",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          0.14
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ -0.944 ] ]
    ]
    Error: [
    	[ [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ -0.944 ] ]
    ]
    Output: [
    	[ [ -0.13216 ] ]
    ]
    Measured: [ [ 0.13999999999986246 ] ]
    Implemented: [ [ 0.14 ] ]
    Error: [ [ -1.375566327510569E-13 ] ]
    Learning Gradient for weight set 0
    Inputs: [
    	[ [ -0.944 ] ]
    ]
    Outputs: [
    	[ [ -0.13216 ] ]
    ]
    Measured Gradient: [ [ -0.9439999999999449 ] ]
    Implemented Gradient: [ [ -0.944 ] ]
    Error: [ [ 5.5067062021407764E-14 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.6312e-14 +- 4.1245e-14 [5.5067e-14 - 1.3756e-13] (2#)
    relativeTol: 2.6022e-13 +- 2.3105e-13 [2.9167e-14 - 4.9127e-13] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.15 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.6821 +- 0.8780 [4.9415 - 10.5328]
    Learning performance: 3.5010 +- 0.1779 [3.1804 - 3.9954]
    
```

