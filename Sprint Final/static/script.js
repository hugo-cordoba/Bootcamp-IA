document.addEventListener("DOMContentLoaded", function() {
    const links = document.querySelectorAll('.sidebar-link');
    const defaultSection = '#generar-imagenes'; // Identificador del div por defecto
    const defaultLink = document.querySelector('a[href="' + defaultSection + '"]'); // Enlace que corresponde al div por defecto

    // Establece por defecto el div de 'GENERAR IMAGENES' como visible
    document.querySelectorAll('.main').forEach(div => div.style.display = 'none'); // Oculta todos los divs primero
    document.querySelector(defaultSection).style.display = 'block'; // Muestra el div de 'GENERAR IMAGENES'
    
    // Remueve la clase activa de todos los enlaces y la añade al enlace de 'GENERAR IMAGENES'
    links.forEach(lnk => lnk.classList.remove('active'));
    defaultLink.classList.add('active'); // Añade la clase 'active' al enlace por defecto

    // Añade los event listeners a los enlaces
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();  // Previene el comportamiento predeterminado de los enlaces
            // Remover la clase activa de todos los links
            links.forEach(lnk => lnk.classList.remove('active'));
            // Añadir la clase activa al link actual
            this.classList.add('active');
            // Ocultar todos los divs
            document.querySelectorAll('.main').forEach(div => div.style.display = 'none');
            // Mostrar el div correspondiente
            const targetDiv = document.querySelector(this.getAttribute('href'));
            targetDiv.style.display = 'block';
        });
    });
});
