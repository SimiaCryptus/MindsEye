# ImgConcatLayer
## BandLimitTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.04 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgConcatLayer",
      "id": "16e60943-f003-439c-918b-b2f300000001",
      "isFrozen": false,
      "name": "ImgConcatLayer/16e60943-f003-439c-918b-b2f300000001",
      "maxBands": 3
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.01 seconds: 
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
    	[ [ 1.376, -0.748 ] ]
    ],
    [
    	[ [ -0.256, -1.484 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.3760000467300415, -0.7480000257492065, -0.25600001215934753 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.06 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ 1.376, -0.748 ] ]
    ],
    [
    	[ [ -0.256, -1.484 ] ]
    ]
    Output: [
    	[ [ 1.3760000467300415, -0.7480000257492065, -0.25600001215934753 ] ]
    ]
    Measured: [ [ 0.9989738464355469, 0.0, 0.0 ], [ 0.0, 1.0001659393310547, 0.0 ] ]
    Implemented: [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ] ]
    Error: [ [ -0.001026153564453125, 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4, 0.0 ] ]
    Feedback for input 1
    Inputs: [
    	[ [ 1.376, -0.748 ] ]
    ],
    [
    	[ [ -0.256, -1.484 ] ]
    ]
    Output: [
    	[ [ 1.3760000467300415, -0.7480000257492065, -0.25600001215934753 ] ]
    ]
    Measured: [ [ 0.0, 0.0, 1.0001659393310547 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented: [ [ 0.0, 0.0, 1.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error: [ [ 0.0, 0.0, 1.659393310546875E-4 ], [ 0.0, 0.0, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1317e-04 +- 2.8201e-04 [0.0000e+00 - 1.0262e-03] (12#)
    relativeTol: 2.2642e-04 +- 2.0288e-04 [8.2963e-05 - 5.1334e-04] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.13 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.6432 +- 1.2842 [4.5540 - 16.2296]
    Learning performance: 1.3700 +- 0.1428 [1.2539 - 2.1772]
    
```

