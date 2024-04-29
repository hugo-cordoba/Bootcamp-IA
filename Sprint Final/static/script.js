document.addEventListener("DOMContentLoaded", function() {
    const links = document.querySelectorAll('.sidebar-link');

    // Inicialización para mostrar la sección por defecto
    setActiveSection('generar-imagenes');

    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = this.getAttribute('href').substring(1);
            setActiveSection(sectionId);
        });
    });

    function setActiveSection(sectionId) {
        // Ocultar todas las secciones
        document.querySelectorAll('.main').forEach(div => div.style.display = 'none');
        // Mostrar la sección seleccionada
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'block';
        } else {
            console.error("Section not found:", sectionId);
        }
        // Activar el enlace correspondiente
        links.forEach(lnk => lnk.classList.remove('active'));
        document.querySelector(`a[href="#${sectionId}"]`).classList.add('active');
    }
});

