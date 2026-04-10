// script.js

// Función para manejar el envío del formulario de contacto
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("contact-form");

  if (form) {
    form.addEventListener("submit", function (event) {
      event.preventDefault();

      const name = document.getElementById("name").value.trim();
      const email = document.getElementById("email").value.trim();
      const message = document.getElementById("message").value.trim();

      if (name === "" || email === "" || message === "") {
        alert("Por favor completa todos los campos.");
        return;
      }

      alert("Gracias por tu mensaje, " + name + ". Nos pondremos en contacto pronto.");

      form.reset();
    });
  }
});
