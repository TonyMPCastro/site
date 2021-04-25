function Carregar(){
var msg = document.getElementById('msg')
var img = document.getElementById('imagem')
var data = new Date()
var hora = data.getHours()

msg.innerHTML = `Agora são ${hora} horas`

if (hora < 5) {
    img.src = 'Bitmap.gif'
    document.body.style.background = '#046a6e'
} else if (hora < 12) {
    img.src = 'Bitmapm.gif'
    document.body.style.background = '#edb690'
} else if (hora < 18) {
    img.src = 'Bitmapt.gif'
    document.body.style.background = '#f79c00'
} else {
    img.src = 'Bitmap.gif'
    document.body.style.background = '#046a6e'
}
}

