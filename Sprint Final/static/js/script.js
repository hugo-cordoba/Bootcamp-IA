document.addEventListener("DOMContentLoaded", function() {
    const links = document.querySelectorAll('.sidebar-link');
    const activeSection = document.body.getAttribute('data-active-section');

    // Establecer la sección activa al cargar
    setActiveSection(activeSection);

    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = this.getAttribute('href').substring(1);  // Obtiene el ID de la sección
            if (sectionId) {
                window.history.pushState({ section: sectionId }, '', '/' + sectionId);  // Cambia la URL sin recargar
                setActiveSection(sectionId);
            }
        });
    });

    window.onpopstate = function(event) {
        // Maneja los cambios en el historial del navegador
        if (event.state) {
            setActiveSection(event.state.section);
        }
    };

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
        // Actualizar enlace activo
        updateActiveLink(sectionId);
    }

    function updateActiveLink(activeId) {
        links.forEach(link => {
            const hrefSubstr = link.getAttribute('href').substring(1);
            if(hrefSubstr === activeId) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
});
