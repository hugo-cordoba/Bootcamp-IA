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

    // Función para copiar al portapapeles
    function copyToClipboard() {
        // Obtén el texto del div hashtagCopy
        var textToCopy = document.getElementById('hashtagCopy').innerText;

        // Crea un elemento temporal textarea
        var tempTextArea = document.createElement('textarea');
        tempTextArea.value = textToCopy;

        // Añade el textarea al documento
        document.body.appendChild(tempTextArea);

        // Selecciona el contenido del textarea
        tempTextArea.select();

        // Copia el texto al portapapeles
        document.execCommand("copy");

        // Remueve el textarea temporal
        document.body.removeChild(tempTextArea);

        //  al usuario que el texto ha sido copiado
        alert("Hashtags copiados al portapapeles");
    }

    const copyButton = document.getElementById('copyButton');
    if (copyButton) {
        copyButton.addEventListener('click', copyToClipboard);
    }

    
});
