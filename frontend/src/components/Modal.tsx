import React, { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    children: React.ReactNode;
    title?: string;
}

export const Modal = ({ isOpen, onClose, children, title }: ModalProps) => {
    const dialogRef = useRef<HTMLDialogElement>(null);

    useEffect(() => {
        const dialog = dialogRef.current;
        if (isOpen) {
            dialog?.showModal();
        } else {
            dialog?.close();
        }
    }, [isOpen]);

    const handleBackdropClick = (e: React.MouseEvent) => {
        if (e.target === dialogRef.current) {
            onClose();
        }
    };

    if (!isOpen) return null;

    return createPortal(
        <dialog
            ref={dialogRef}
            className="modal-dialog"
            onClick={handleBackdropClick}
            onCancel={onClose}
        >
            <div className="modal-content">
                <header className="modal-header">
                    {title && <h2>{title}</h2>}
                    <button className="close-button" onClick={onClose}>
                        ×
                    </button>
                </header>
                <div className="modal-body">{children}</div>
            </div>
        </dialog>,
        document.body
    );
};
