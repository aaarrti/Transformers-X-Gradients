## `Transformers` X `Gradients`

like GradientXInput, pun intended.

Collection of gradient-based XAI methods for TensorFlow models from HuggingFace Hub.

Work in Progress!!!


<style>
    .container {
        line-height: 1.4;
        text-align: center;
        margin: 10px 10px 10px 10px;
        color: black;
        background: white;
    }
    p {
        font-size: 16px;
    }
    .highlight-container, .highlight {
        position: relative;
        border-radius: 10% 10% 10% 10%;
    }
    .highlight-container {
        display: inline-block;
    }
    .highlight-container:before {
        content: " ";
        display: block;
        height: 90%;
        width: 100%;
        margin-left: -3px;
        margin-right: -3px;
        position: absolute;
        top: -1px;
        left: -1px;
        padding: 10px 3px 3px 10px;
    }
</style>
<div class="container">
    <p>
        Gradient X Input <br>
        <span class="highlight-container" style="background:rgb(0.0,0.0,255.0);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(230.14891052246094,230.14891052246094,255.0);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,223.685546875,223.685546875);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(239.6996307373047,239.6996307373047,255.0);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(233.97622680664062,233.97622680664062,255.0);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(236.8782196044922,236.8782196044922,255.0);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,250.2151336669922,250.2151336669922);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(234.85159301757812,234.85159301757812,255.0);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(206.59451293945312,206.59451293945312,255.0);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(235.87474060058594,235.87474060058594,255.0);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(237.4338836669922,237.4338836669922,255.0);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(202.831298828125,202.831298828125,255.0);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(176.82305908203125,176.82305908203125,255.0);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,15.641389846801758,15.641389846801758);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,215.24859619140625,215.24859619140625);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(224.62310791015625,224.62310791015625,255.0);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(198.2104949951172,198.2104949951172,255.0);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(235.98031616210938,235.98031616210938,255.0);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(235.90567016601562,235.90567016601562,255.0);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,189.1565399169922,189.1565399169922);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(217.50811767578125,217.50811767578125,255.0);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>

</div>

<div class="container">
    <p>
        Integrated Gradients <br>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,97.97626495361328,97.97626495361328);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,137.02178955078125,137.02178955078125);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,103.36837768554688,103.36837768554688);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,145.1651611328125,145.1651611328125);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,115.20925903320312,115.20925903320312);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,110.63146209716797,110.63146209716797);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,70.82160186767578,70.82160186767578);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,139.3142547607422,139.3142547607422);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,105.95587158203125,105.95587158203125);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,87.99478149414062,87.99478149414062);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,140.76976013183594,140.76976013183594);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,99.2268295288086,99.2268295288086);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,96.96973419189453,96.96973419189453);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,125.80677795410156,125.80677795410156);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,29.600852966308594,29.600852966308594);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,150.7434539794922,150.7434539794922);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,146.2736358642578,146.2736358642578);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,126.10820770263672,126.10820770263672);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,117.285400390625,117.285400390625);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,60.55534362792969,60.55534362792969);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,93.12007141113281,93.12007141113281);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>

### Roadmap
- NLP tasks
  - text classification
  - QA
  - token classification
  - fill mask
  - text completion ???
- Libraries
  - HF
  - TensorRT
  - TF models official
  - keras NLP ?
