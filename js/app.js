const dropdownBtn = document.getElementById('dropdownBtn');
const dropdownContent = document.getElementById('dropdownContent');

dropdownBtn.addEventListener('click', () => {
    dropdownContent.style.display = dropdownContent.style.display === 'block' ? 'none' : 'block';
});

window.addEventListener('click', e => {
    if (!dropdownBtn.contains(e.target) && !dropdownContent.contains(e.target)) {
        dropdownContent.style.display = 'none';
    }
});


// document.querySelectorAll('.bibtex-entry').forEach(entry => {
//   const button = document.createElement('button');
//   button.textContent = 'Copy BibTeX';
//   button.style.marginBottom = '4px';
  
//   button.addEventListener('click', () => {
//     navigator.clipboard.writeText(entry.textContent).then(() => {
//       button.textContent = 'Copied!';
//       setTimeout(() => button.textContent = 'Copy BibTeX', 1500);
//     });
//   });

//   entry.parentNode.insertBefore(button, entry);
// });
