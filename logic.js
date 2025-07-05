//canvas
let board;
let bHeight = 200;
let bWidth = 700
let context;

//dinosaur
let dWidth = 64;
let dHeight = 70;
let dX = 75;
let dY = bHeight - dHeight;
let dImg;

let dino = {
    x : dX,
    y : dY,
    width: dWidth,
    height: dHeight
}

//cactus
let array = [];
let c1w = 24;
let c1i;
let c2w = 49;
let c2i;
let c3w = 72;
let c3i;

let ch = 25;
let cx = 700;
let cy = bHeight - ch;

let vX = -7;
let vY = 0;
let gravity = 0.6;

let gameOver = false;
let score = 0;

window.onload = function() {
    board = document.getElementById("canvas");
    board.height = bHeight;
    board.width = bWidth;

    context = board.getContext("2d");

    dImg = new Image();
    dImg.src = "./img/dino.png";
    dImg.onload = function() {
        context.drawImage(dImg, dino.x, dino.y, dino.width, dino.height);
    }

    c1i = new Image();
    c1i.src = "./img/cactus1.png"
    
    c2i = new Image();
    c2i.src = "./img/cactus2.png"

    c3i = new Image();
    c3i.src = "./img/cactus3.png"

    requestAnimationFrame(u);

    s = (Math.random() + 1) * 1000;
    setInterval(placeCactus, s);
    document.addEventListener("keydown", moveDino);
}

function u(){
    requestAnimationFrame(u);
    if(gameOver){
        return
    }

    context.clearRect(0, 0, bWidth, bHeight);

    vX -= 0.01
    vY += gravity;
    score++;


    dino.y = Math.min(dino.y+vY, dY);
    context.drawImage(dImg, dino.x, dino.y, dino.width, dino.height);
    context.font = "18px Arial";
    context.fillText(score,10, 25)

    for (let i=0; i<array.length; i++){
        let c = array[i]
        c.x += vX;
        context.drawImage(c.img, c.x, c.y, c.width, c.height);

        if (crash(dino, c)) {
            gameOver = true;
            dImg.src = "./img/dino-dead.png";
            dImg.onload = function() {
                context.drawImage(dImg, dino.x, dino.y, dino.width, dino.height);
            }
        }
    }
}

function moveDino(e){
    if(gameOver){
        return
    }

    if ((e.code == "Space" || e.code == "ArrowUp") && dino.y == dY){
        vY = -10;
    }
}

function placeCactus(){
    if(gameOver){
        return
    }

    let cactus = {
        img : null,
        x : cx,
        y : cy,
        width : null,
        height: ch,
    }

    let cactusChance = Math.random();

    if(cactusChance > 0.9){
        cactus.img = c1i;
        cactus.width = c1w;
        cactus.height = ch;
        array.push(cactus);
        console.log("placed cactus");
    } else if(cactusChance > 0.5){
        cactus.img = c2i;
        cactus.width = c2w;
        cactus.height = ch;
        array.push(cactus);
    } else{
        cactus.img = c3i;
        cactus.width = c3w;
        cactus.height = ch;
        array.push(cactus);
    }

    if(array.length > 5){
        array.shift();
    }
}

let crash = (a, b) => {
    const tolerance = 5;
    return  a.x < b.x + b.width - tolerance &&
            a.x + a.width > b.x + tolerance &&
            a.y < b.y + b.height - tolerance &&
            a.y + a.height > b.y + tolerance;
}