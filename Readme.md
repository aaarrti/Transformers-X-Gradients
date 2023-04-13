## `Transformers` X `Gradients`

like GradientXInput, pun intended.

Collection of gradient-based XAI methods for TensorFlow models from HuggingFace Hub.

Work in Progress!!!

<div>
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
    <p> Gradient Norm <br>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,190.44862365722656,190.44862365722656);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,212.59649658203125,212.59649658203125);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.2554168701172,199.2554168701172);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,216.51776123046875,216.51776123046875);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,197.98631286621094,197.98631286621094);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.52127075195312,199.52127075195312);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,173.62420654296875,173.62420654296875);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.695068359375,199.695068359375);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,181.93450927734375,181.93450927734375);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,168.0614776611328,168.0614776611328);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,200.93209838867188,200.93209838867188);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,178.53001403808594,178.53001403808594);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,172.0614013671875,172.0614013671875);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,183.44143676757812,183.44143676757812);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,103.0122299194336,103.0122299194336);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,197.0795440673828,197.0795440673828);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,192.9103240966797,192.9103240966797);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,173.73768615722656,173.73768615722656);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,166.02821350097656,166.02821350097656);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,123.84181213378906,123.84181213378906);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,164.94869995117188,164.94869995117188);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>
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
<div class="container">
    <p>
        Smooth Grad + Gradient Norm <br>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,190.6959228515625,190.6959228515625);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,212.80825805664062,212.80825805664062);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.57901000976562,199.57901000976562);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,216.68418884277344,216.68418884277344);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,198.28843688964844,198.28843688964844);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.55279541015625,199.55279541015625);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,173.8881072998047,173.8881072998047);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.83154296875,199.83154296875);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,182.0014190673828,182.0014190673828);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,168.2892303466797,168.2892303466797);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,200.9700164794922,200.9700164794922);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,178.50901794433594,178.50901794433594);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,172.2283172607422,172.2283172607422);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,183.904296875,183.904296875);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,102.8530502319336,102.8530502319336);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,197.36643981933594,197.36643981933594);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,193.03028869628906,193.03028869628906);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,173.68768310546875,173.68768310546875);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,166.13414001464844,166.13414001464844);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,124.19740295410156,124.19740295410156);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,164.96923828125,164.96923828125);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>

<div class="container">
    <p>
        Noise Grad + Gradient Norm <br>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,189.19313049316406,189.19313049316406);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,212.06800842285156,212.06800842285156);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,198.5439910888672,198.5439910888672);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,216.9097442626953,216.9097442626953);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,197.43055725097656,197.43055725097656);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.1936492919922,199.1936492919922);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,170.92037963867188,170.92037963867188);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,198.1800079345703,198.1800079345703);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,178.603515625,178.603515625);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,164.21615600585938,164.21615600585938);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.00425720214844,199.00425720214844);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,177.22653198242188,177.22653198242188);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,170.9322509765625,170.9322509765625);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,185.05380249023438,185.05380249023438);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,106.45462799072266,106.45462799072266);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,198.1622772216797,198.1622772216797);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,193.64308166503906,193.64308166503906);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,176.4576873779297,176.4576873779297);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,166.50045776367188,166.50045776367188);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,124.95288848876953,124.95288848876953);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,166.07656860351562,166.07656860351562);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>

</div>

<div class="container">
    <p>
        NoiseGrad++ + Gradient Norm <br>
        <span class="highlight-container" style="background:rgb(255.0,143.1840057373047,143.1840057373047);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,184.0985107421875,184.0985107421875);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,187.45346069335938,187.45346069335938);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,186.86337280273438,186.86337280273438);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,179.39414978027344,179.39414978027344);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,192.878662109375,192.878662109375);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,187.46568298339844,187.46568298339844);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,182.25546264648438,182.25546264648438);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,151.06480407714844,151.06480407714844);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,177.48504638671875,177.48504638671875);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,170.0495147705078,170.0495147705078);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,198.44134521484375,198.44134521484375);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,172.0955047607422,172.0955047607422);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,183.46829223632812,183.46829223632812);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,181.69700622558594,181.69700622558594);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,168.32305908203125,168.32305908203125);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,171.58189392089844,171.58189392089844);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,150.7019805908203,150.7019805908203);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,144.99021911621094,144.99021911621094);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,146.93807983398438,146.93807983398438);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,174.5519561767578,174.5519561767578);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>

<div class="container">
    <p>
        Smooth Grad + Gradient X Input <br>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,187.52066040039062,187.52066040039062);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,209.1520538330078,209.1520538330078);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,193.63079833984375,193.63079833984375);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,215.96192932128906,215.96192932128906);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,194.1007080078125,194.1007080078125);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,197.724853515625,197.724853515625);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,166.1660919189453,166.1660919189453);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,197.0251922607422,197.0251922607422);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,173.65452575683594,173.65452575683594);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,158.0315704345703,158.0315704345703);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,197.5176544189453,197.5176544189453);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,177.2938232421875,177.2938232421875);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,172.2130889892578,172.2130889892578);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,191.694091796875,191.694091796875);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,117.76380920410156,117.76380920410156);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,201.1962127685547,201.1962127685547);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,196.65875244140625,196.65875244140625);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,181.15008544921875,181.15008544921875);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,170.701416015625,170.701416015625);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,128.9667205810547,128.9667205810547);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,168.354736328125,168.354736328125);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>

<div class="container">
    <p>
        Noise Grad + Gradient X Input <br>
        <span class="highlight-container" style="background:rgb(0.0,0.0,255.0);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(233.50721740722656,233.50721740722656,255.0);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,227.3994903564453,227.3994903564453);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(249.64120483398438,249.64120483398438,255.0);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(241.16929626464844,241.16929626464844,255.0);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(233.90621948242188,233.90621948242188,255.0);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,214.82528686523438,214.82528686523438);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(231.13624572753906,231.13624572753906,255.0);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(199.0653839111328,199.0653839111328,255.0);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(224.93092346191406,224.93092346191406,255.0);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(220.75753784179688,220.75753784179688,255.0);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(204.4977569580078,204.4977569580078,255.0);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(160.79237365722656,160.79237365722656,255.0);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,158.57225036621094,158.57225036621094);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(253.31103515625,253.31103515625,255.0);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(212.01361083984375,212.01361083984375,255.0);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(196.45521545410156,196.45521545410156,255.0);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(231.3936309814453,231.3936309814453,255.0);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(236.98036193847656,236.98036193847656,255.0);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,214.17872619628906,214.17872619628906);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(213.25140380859375,213.25140380859375,255.0);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>

<div class="container">
    <p>
        NoiseGrad++ + Gradient X Input <br>
        <span class="highlight-container" style="background:rgb(255.0,134.1307373046875,134.1307373046875);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,186.99171447753906,186.99171447753906);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,187.5166015625,187.5166015625);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,185.3190155029297,185.3190155029297);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,177.40126037597656,177.40126037597656);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,189.19143676757812,189.19143676757812);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,186.23684692382812,186.23684692382812);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,181.71728515625,181.71728515625);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,150.83946228027344,150.83946228027344);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,175.00839233398438,175.00839233398438);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,169.73931884765625,169.73931884765625);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,193.98178100585938,193.98178100585938);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,160.2654571533203,160.2654571533203);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,178.9566192626953,178.9566192626953);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,176.32456970214844,176.32456970214844);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,167.6029815673828,167.6029815673828);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,170.0122833251953,170.0122833251953);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,155.51817321777344,155.51817321777344);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,148.7647247314453,148.7647247314453);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,143.005859375,143.005859375);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,173.65208435058594,173.65208435058594);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>

<div class="container">
    <p>
        Smooth Grad + Integrated Gradients <br>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,187.89675903320312,187.89675903320312);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,210.2369384765625,210.2369384765625);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,195.111572265625,195.111572265625);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,215.10464477539062,215.10464477539062);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,196.41677856445312,196.41677856445312);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,198.47323608398438,198.47323608398438);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,163.1812744140625,163.1812744140625);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.12478637695312,199.12478637695312);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,175.4522705078125,175.4522705078125);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,154.05384826660156,154.05384826660156);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,199.95834350585938,199.95834350585938);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,185.12686157226562,185.12686157226562);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,176.14352416992188,176.14352416992188);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,193.9889678955078,193.9889678955078);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,123.84754180908203,123.84754180908203);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,202.184326171875,202.184326171875);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,197.02407836914062,197.02407836914062);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,186.30194091796875,186.30194091796875);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,173.63235473632812,173.63235473632812);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,125.22667694091797,125.22667694091797);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,170.42691040039062,170.42691040039062);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>

<div class="container">
    <p>
        Noise Grad + Integrated Gradients <br>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,111.7264404296875,111.7264404296875);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,144.50714111328125,144.50714111328125);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,114.30277252197266,114.30277252197266);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,156.2357177734375,156.2357177734375);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,122.1633529663086,122.1633529663086);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,117.0184326171875,117.0184326171875);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,59.93711853027344,59.93711853027344);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,144.34535217285156,144.34535217285156);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,105.24752807617188,105.24752807617188);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,70.15374755859375,70.15374755859375);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,152.44393920898438,152.44393920898438);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,120.3367919921875,120.3367919921875);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,109.26026153564453,109.26026153564453);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,140.349853515625,140.349853515625);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,57.37871551513672,57.37871551513672);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,162.97544860839844,162.97544860839844);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,159.30250549316406,159.30250549316406);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,145.12838745117188,145.12838745117188);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,129.734130859375,129.734130859375);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,67.79318237304688,67.79318237304688);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,105.54365539550781,105.54365539550781);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>

<div class="container">
    <p>
        NoiseGrad++ + Integrated Gradients <br>
        <span class="highlight-container" style="background:rgb(255.0,124.97313690185547,124.97313690185547);">
            <span class="highlight"> [CLS] </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,184.70091247558594,184.70091247558594);">
            <span class="highlight"> like </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,186.0247344970703,186.0247344970703);">
            <span class="highlight"> four </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,180.6410369873047,180.6410369873047);">
            <span class="highlight"> times </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,172.48988342285156,172.48988342285156);">
            <span class="highlight"> a </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,186.69883728027344,186.69883728027344);">
            <span class="highlight"> year </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,181.195068359375,181.195068359375);">
            <span class="highlight"> i </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,178.42311096191406,178.42311096191406);">
            <span class="highlight"> red </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,143.38201904296875,143.38201904296875);">
            <span class="highlight"> ##is </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,169.5498809814453,169.5498809814453);">
            <span class="highlight"> ##co </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,160.8484344482422,160.8484344482422);">
            <span class="highlight"> ##ver </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,188.75741577148438,188.75741577148438);">
            <span class="highlight"> b </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,144.67234802246094,144.67234802246094);">
            <span class="highlight"> ##jo </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,169.0419158935547,169.0419158935547);">
            <span class="highlight"> ##rk </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,171.87969970703125,171.87969970703125);">
            <span class="highlight"> and </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,160.5388641357422,160.5388641357422);">
            <span class="highlight"> listen </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,170.9183349609375,170.9183349609375);">
            <span class="highlight"> to </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,154.4762420654297,154.4762420654297);">
            <span class="highlight"> her </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,146.69021606445312,146.69021606445312);">
            <span class="highlight"> full </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,143.18008422851562,143.18008422851562);">
            <span class="highlight"> disco </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,172.32814025878906,172.32814025878906);">
            <span class="highlight"> ##graphy </span>
        </span>
        <span class="highlight-container" style="background:rgb(255.0,0.0,0.0);">
            <span class="highlight"> [SEP] </span>
        </span>
    </p>
</div>
</div>
